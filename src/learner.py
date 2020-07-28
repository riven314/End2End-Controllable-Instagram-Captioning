import os
import json
import time

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence

from src.utils import AverageMeter
from src.utils import adjust_learning_rate, accuracy, save_checkpoint, clip_gradient

from ranger import Ranger

class Learner:
    def __init__(self, encoder, decoder, train_loader, val_loader, test_loader, cfg):
        assert cfg.device in ['cpu', 'cuda']

        self._load_word_maps(cfg.word_map_file)

        self.data_name = cfg.data_name
        self.save_dir = cfg.save_dir
        os.makedirs(cfg.save_dir, exist_ok = True)
        
        self.optimizer = cfg.optimizer
        self.encoder = encoder
        self.decoder = decoder
        self.fine_tune_encoder = cfg.fine_tune_encoder
        self.grad_clip = cfg.grad_clip
        self.alpha_c = cfg.alpha_c
        self.confidence_c = cfg.confidence_c

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.best_bleu4 = 0.
        self.start_epoch = 0
        self.epochs_since_improvement = 0

        self.epochs = cfg.epochs
        self.tolerance_epoch = cfg.tolerance_epoch
        self.adjust_epoch = cfg.adjust_epoch
        self.adjust_step = cfg.adjust_step
        self.decay_epoch = cfg.decay_epoch
        self.decay_step = cfg.decay_step

        self.print_freq = cfg.print_freq

        # load pretrained word embedding weight (optional)
        if cfg.word_embedding_weight is not None:
            self._load_word_embedding(cfg.word_embedding_weight)

        # set up optimizer
        self._init_optimizer(cfg.encoder_lr, cfg.decoder_lr)
        if cfg.checkpoint_file is not None:
            self._load_checkpoint(cfg.checkpoint_file)

        self.device = torch.device(cfg.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def main_run(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch

            if (epoch != 0) and (epoch % self.decay_epoch == 0):
                adjust_learning_rate(self.decoder_optimizer, self.decay_step)
                if self.fine_tune_encoder:
                    adjust_learning_rate(self.encoder_optimizer, self.decay_step)

            if self.epochs_since_improvement == self.tolerance_epoch:
                break
            if (self.epochs_since_improvement > 0) and (self.epochs_since_improvement % self.adjust_epoch == 0):
                adjust_learning_rate(self.decoder_optimizer, self.adjust_step)
                if self.fine_tune_encoder:
                    adjust_learning_rate(self.encoder_optimizer, self.adjust_step)

            self.train_one_epoch()

            current_bleu4 = self.val_one_epoch()
            is_best = current_bleu4 > self.best_bleu4
            if not is_best:
                self.epochs_since_improvement += 1
                print(f"\nEpochs since last improvement: {self.epochs_since_improvement}")
            else:
                self.epochs_since_improvement = 0

            if self.test_loader is not None:
                self.test_one_epoch()
            
            self._save_checkpoint(current_bleu4, is_best)

    def train_one_epoch(self):
        self.decoder.train()  # train mode (dropout and batchnorm is used)
        self.encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy
        
        start = time.time()
        log_softmax = nn.LogSoftmax()

        # Batches
        for i, (imgs, caps, caplens, _, len_class, img_ids) in enumerate(self.train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)
            len_class = len_class.to(self.device)

            # Forward prop.
            imgs = self.encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens, len_class)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = self.criterion(scores.data, targets.data)

            # Add confidence penalty
            if self.confidence_c is not None:
                tgt_batch_idx = scores.batch_sizes[:3].sum() # only consider predictions for first 3 words
                probs = log_softmax(scores.data[:tgt_batch_idx]).exp()
                entropies = Categorical(probs = probs).entropy()
                loss -= self.confidence_c * entropies.mean()

            # Add doubly stochastic attention regularization
            loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            self.decoder_optimizer.zero_grad()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if self.grad_clip is not None:
                clip_gradient(self.decoder_optimizer, self.grad_clip)
                if self.encoder_optimizer is not None:
                    clip_gradient(self.encoder_optimizer, self.grad_clip)

            # Update weights
            self.decoder_optimizer.step()
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores.data, targets.data, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                        self.current_epoch, i, len(self.train_loader),
                        batch_time = batch_time,
                        data_time = data_time, loss = losses,
                        top5 = top5accs
                    ))

    def val_one_epoch(self):
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        if self.encoder is not None:
            self.encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        log_softmax = nn.LogSoftmax()
        start = time.time()

        # references (true captions) for calculating BLEU-4 score
        # hypotheses (predictions)
        references, hypotheses = list(), list()  

        # solves the issue #57
        with torch.no_grad():
            # Batches
            for i, (imgs, caps, caplens, allcaps, len_class, img_ids) in enumerate(self.val_loader):

                # Move to device, if available
                imgs = imgs.to(self.device)
                caps = caps.to(self.device)
                caplens = caplens.to(self.device)
                len_class = len_class.to(self.device)

                # Forward prop.
                if self.encoder is not None:
                    imgs = self.encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens, len_class)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss = self.criterion(scores.data, targets.data)

                # Add confidence penalty
                if self.confidence_c is not None:            
                    tgt_batch_idx = scores.batch_sizes[:3].sum() # only consider predictions for first 3 words
                    probs = log_softmax(scores.data[:tgt_batch_idx]).exp()
                    entropies = Categorical(probs = probs).entropy()
                    loss -= self.confidence_c * entropies.mean()

                # Add doubly stochastic attention regularization
                loss += self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores.data, targets.data, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                            i, len(self.val_loader), batch_time = batch_time,
                            loss = losses, top5 = top5accs
                        ))

                # References
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim = 2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss = losses, top5 = top5accs, bleu = bleu4
            ))
        return bleu4

    def test_one_epoch(self):
        pass

    def _init_optimizer(self, encoder_lr, decoder_lr):
        assert self.optimizer in ['adam', 'ranger']

        optimizer = Adam if self.optimizer == 'adam' else Ranger
        print(f'optimizer used: {optimizer}')

        self.decoder_optimizer = optimizer(params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
                                           lr = decoder_lr)

        if self.fine_tune_encoder:
            self.encoder_optimizer = optimizer(params = filter(lambda p: p.requires_grad, self.encoder.parameters()),
                                               lr = encoder_lr)
        else:
            self.encoder_optimizer = None

    def _save_checkpoint(self, current_bleu4, is_best):
        save_checkpoint(self.data_name, self.current_epoch, self.epochs_since_improvement, 
                        self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer, 
                        self.save_dir, current_bleu4, is_best)

    def _load_checkpoint(self, checkpoint):
        assert os.path.isfile(checkpoint)
        checkpoint_dict = torch.load(checkpoint)

        self.best_bleu4 = checkpoint_dict['bleu-4']
        self.start_epoch = checkpoint_dict['epoch'] + 1
        self.epochs_since_improvement = checkpoint_dict['epochs_since_improvement']

        self.decoder.load_state_dict(checkpoint_dict['decoder'])
        self.decoder_optimizer.load_state_dict(checkpoint_dict['decoder_optimizer'])
        decoder_lr = self.decoder_optimizer.param_groups[0]['lr']

        if (checkpoint_dict['encoder_optimizer'] is not None) and self.fine_tune_encoder:
            self.encoder.load_state_dict(checkpoint_dict['encoder'])
            self.encoder_optimizer.load_state_dict(checkpoint_dict['encoder_optimizer'])
            encoder_lr = self.encoder_optimizer.param_groups[0]['lr']
        elif (checkpoint_dict['encoder_optimizer'] is None) and self.fine_tune_encoder:
            encoder_lr = self.encoder_optimizer.param_groups[0]['lr']
        else:
            encoder_lr = None

        self._init_optimizer(encoder_lr, decoder_lr)

        print('picked up learning rate from checkpoint \n')
        print(f'encoder lr: {encoder_lr}')
        print(f'decoder lr: {decoder_lr}')

    def _load_word_embedding(self, word_embedding_weight):
        assert word_embedding_weight
        weight = np.load(word_embedding_weight)
        self.decoder.load_pretrained_embeddings(weight)
        print(f'pretrained word embedding loaded: {word_embedding_weight}')
    
    def _load_word_maps(self, word_map_file):
        assert os.path.isfile(word_map_file)

        with open(word_map_file, 'r') as f:
            self.word_map = json.load(f)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        
