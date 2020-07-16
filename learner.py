import os
import json

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence

from utils import AverageMeter
from utils import adjust_learning_rate, accuracy, save_checkpoint


class Leaner:
    def __init__(self, encoder, decoder, train_loader, val_loader, test_loader, cfg):
        self._load_word_maps(cfg.word_map_file)

        self.encoder = encoder
        self.decoder = decoder
        self.fine_tune_encoder = cfg.fine_tune_encoder

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.best_bleu4 = 0.
        self.start_epoch = 0
        self.epochs_since_improvement = 0

        self._init_optimizer(cfg.encoder_lr, cfg.decoder_lr, cfg.fine_tune_encoder)
        if cfg.checkpoint is not None:
            self._load_checkpoint(cfg.checkpoint)

        self.device = cfg.device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.epochs = self.epochs
        self.tolerance_epoch = cfg.tolerance_epoch
        self.adjust_epoch = cfg.adjust_epoch
        self.adjust_step = cfg.adjust_step
        self.print_freq = cfg.print_freq

    def main_run(self):
        for epoch in range(self.start_epoch, self.epochs):

            if self.epochs_since_improvement == self.tolerance_epoch:
                break
            if (self.epochs_since_improvement > 0) and (self.epochs_since_improvement % self.adjust_epoch == 0):
                adjust_learning_rate(self.decoder_optimizer, self.adjust_step)
                if self.fine_tune_encoder:
                    adjust_learning_rate(self.encoder_optimizer, self.adjust_step)

            self.train_one_epoch()

            recent_bleu4 = self.val_one_epoch()
            is_best = recent_bleu4 > self.best_bleu4
            if not is_best:
                self.epochs_since_improvement += 1
                print(f"\nEpochs since last improvement: {self.epochs_since_improvement}")
            else:
                self.epochs_since_improvement = 0

            self.test_one_epoch()
            self._save_checkpoint()

    def train_one_epoch(self):
        pass

    def val_one_epoch(self):
        bleu4_score = 0
        return bleu4_score

    def test_one_epoch(self, n):
        pass

    def _init_optimizer(self, encoder_lr, decoder_lr):
        self.decoder_optimizer = Adam(
            params = filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr = decoder_lr
        )

        if self.fine_tune_encoder:
            self.encoder_optimizer = Adam(
                params = filter(lambda p: p.requires_grad, self.encoder.parameters()), 
                lr = encoder_lr
            )
        else:
            self.encoder_optimizer = None

    def _save_checkpoint(self):
        pass

    def _load_checkpoint(self, checkpoint):
        assert os.path.isfile(checkpoint)
        checkpoint_dict = torch.load(checkpoint)

        self.best_bleu4 = checkpoint_dict['bleu-4']
        self.start_epoch = checkpoint_dict['epoch'] + 1
        self.epochs_since_improvement = checkpoint_dict['epochs_since_improvement']

        self.encoder_optimizer.load_state_dict(checkpoint_dict['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint_dict['decoder_optimizer'])
        encoder_lr = self.encoder_optimizer.param_groups[0]['lr']
        decoder_lr = self.decoder_optimizer.param_groups[0]['lr']
        self._init_optimizer(encoder_lr, decoder_lr)

        print('picked up learning rate from checkpoint \n')
        print(f'encoder lr: {encoder_lr}')
        print(f'decoder lr: {decoder_lr}')
    
    def _load_word_maps(self, word_map_file):
        assert os.path.isfile(word_map_file)

        with open(word_map_file, 'r') as f:
            self.word_map = json.load(f)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}
        
        

