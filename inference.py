""" run trained model on test set, get gt captions v.s. predicted captions """
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import emoji
from transformers import AutoTokenizer

from data.utils import read_json
from src.models import get_encoder_decoder
from src.datasets import CaptionDataset
from src.utils import *
from src.word_map_utils import get_wp_tokenizer


checkpoint_dir = './ckpts/v14_wstyle_wp_full_entropy_3.0'
data_folder = './data/meta_wstyle/data_mid_clean_wonumber_wp'
data_name = 'flickr8k_1_cap_per_img_1_min_word_freq'
checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth')
word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'


word_map = read_json(word_map_file)
rev_word_map = {v: k for k, v in word_map.items()}
emoji_set = [w for w in word_map_file.keys() if w.startswith(':') and w.endswith(':')]
vocab_size = len(word_map)

cfg_path = os.path.join(checkpoint_dir, 'config.json')
cfg = read_json(cfg_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder, decoder = get_encoder_decoder(cfg)
checkpoint = torch.load(checkpoint_file)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()


def run_test_per_beamsize_style(beam_size, length_class, is_emoji,
                                data_type = 'TEST', n = -1, subword = False):

    assert data_type in ['TRAIN', 'VAL', 'TEST']
    assert length_class in [0, 1, 2]

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    tfms = transforms.Compose([normalizer])
    dataset = CaptionDataset(data_folder, data_name, data_type, transform = tfms)
    dataloader = DataLoader(dataset, batch_size = 1, 
                            shuffle = False, num_workers = 1, 
                            pin_memory = True)
    
    tokenizer = None
    if subword:
        tokenizer = get_wp_tokenizer(data = None, emoji_set = emoji_set)

    len_class = torch.as_tensor([length_class]).long().to(device)
    is_emoji = torch.as_tensor([is_emoji]).long().to(device)

    results = []
    for i, (image, caps, _, allcaps, style_dicts, img_ids) in enumerate(tqdm(dataloader)):
        
        if i == n:
            break

        k = beam_size

        # move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        len_class_embedding = decoder.length_class_embedding(len_class)
        is_emoji_embedding = decoder.is_emoji_embedding(is_emoji)
        
        style_embed_dim = len_class_embedding.size(-1)
        len_class_embedding = len_class_embedding.expand(k, style_embed_dim)
        is_emoji_embedding = is_emoji_embedding.expand(k, style_embed_dim)

        # Start decoding
        step = 1
        with torch.no_grad():
            h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        with torch.no_grad():
            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                # (s, decoder_dim)
                h, c = decoder.decode_step(
                    torch.cat([embeddings, len_class_embedding, is_emoji_embedding,  awe], dim = 1),
                    (h, c)) 

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim = 1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

        # handle corner case when prediction is failed (empty, or too long)
        if len(complete_seqs_scores) == 0:
            print(f'{img_ids[0]} has no complete sentence')
            img_cap = caps.tolist()[0]
            img_caption = [w for w in img_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            img_caption = [rev_word_map[s] for s in img_caption]
            
            if subword:
                assert tokenizer is not None
                ref_enc = tokenizer.convert_tokens_to_ids(img_caption)
                img_caption = tokenizer.decode(ref_enc)
            else:
                img_caption = ' '.join(img_caption)
            
            gt_len_class = int(style_dicts['length_class'].cpu().squeeze())
            gt_is_emoji = int(style_dicts['is_emoji'].cpu().squeeze())
            result = {
                'img_id': img_ids[0], 'gt_length_class': gt_len_class, 'gt_is_emoji': gt_is_emoji, 
                'data_type': data_type, 'gt_caption': img_caption, 
                f'length_class_len{length_class:02}_emoji{is_emoji:02}': 'NA'
                }
            results.append(result)
            continue

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # references & prediction
        img_cap = caps.tolist()[0]
        img_caption = [w for w in img_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        predict = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        
        if subword:
            assert tokenizer is not None
            img_caption = [rev_word_map[s] for s in img_caption]
            ref_enc = tokenizer.convert_tokens_to_ids(img_caption)
            img_caption = tokenizer.decode(ref_enc)

            predict = [rev_word_map[s] for s in predict]
            pred_enc = tokenizer.convert_tokens_to_ids(predict)
            predict = tokenizer.decode(pred_enc)            
        else:
            img_caption = ' '.join([rev_word_map[s] for s in img_caption])
            predict = ' '.join([rev_word_map[s] for s in predict])

        gt_len_class = int(style_dicts['length_class'].cpu().squeeze())
        gt_is_emoji = int(style_dicts['is_emoji'].cpu().squeeze())
        result = {
            'img_id': img_ids[0], 'data_type': data_type,
            'gt_length_class': gt_len_class, 'gt_is_emoji': gt_is_emoji, 'gt_caption': img_caption, 
            f'length_class_{length_class:02}_emoji_class_{is_emoji:02}': predict
            }
        results.append(result)
    return results


if __name__ == '__main__':
    beam_size = 10
    for data_type in ['TEST', 'TRAIN', 'VAL']:
        result_csv = os.path.join(checkpoint_dir, f'benchmarks_{data_type.lower()}.csv')
        
        agg_results = []
        for len_class in [0, 1, 2]:
            for emoji_class in [0, 1]:
                print(f'data_type: {data_type}, beam size: {beam_size}, length class: {len_class}, emoji class: {emoji_class}')
                results = run_test_per_beamsize_style(
                        beam_size, 
                        length_class = len_class, is_emoji = emoji_class,
                        data_type = data_type, n = 200, subword = True
                        )

                if agg_results == []:
                    agg_results = results
                else:
                    for i in range(len(agg_results)):
                        agg_results[i].update(results[i])

        result_df = pd.DataFrame(agg_results)
        result_df.to_csv(result_csv, index = False)
        print(f'result csv written: {result_csv}')
