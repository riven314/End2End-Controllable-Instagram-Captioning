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

from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = 'data/meta_wostyle/data_full'
data_name = 'flickr8k_1_cap_per_img_5_min_word_freq'
checkpoint_file = './ckpts/BEST_checkpoint_flickr8k_1_cap_per_img_5_min_word_freq.pth'
word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

attention_dim = 512
emb_dim = 512
decoder_dim = 512
vocab_size = len(word_map) 
dropout = 0.5

encoder = Encoder()
decoder = DecoderWithAttention(attention_dim = attention_dim,
                               embed_dim = emb_dim,
                               decoder_dim = decoder_dim,
                               vocab_size = len(word_map),
                               dropout = dropout)

checkpoint = torch.load(checkpoint_file)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()


def run_test_per_beamsize_style(beam_size, data_type = 'TEST', n = -1):
    assert data_type in ['TRAIN', 'VAL', 'TEST']

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    tfms = transforms.Compose([normalizer])
    dataset = CaptionDataset(data_folder, data_name, data_type, transform = tfms)
    dataloader = DataLoader(dataset, batch_size = 1, 
                            shuffle = False, num_workers = 1, 
                            pin_memory = True)

    results = []
    for i, (image, caps, _, allcaps, img_ids) in enumerate(tqdm(dataloader)):

        if i == n:
            break

        image = image.to(device)  # (1, 3, 256, 256)

        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        predict = decoder.beam_search(encoder_out, word_map, k = beam_size)
        predict = ' '.join([rev_word_map[s] for s in predict])

        # references
        img_cap = caps.tolist()[0]
        img_caption = [w for w in img_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        img_caption = ' '.join([rev_word_map[s] for s in img_caption])

        result = {'img_id': img_ids[0], 'data_type': data_type, 'gt_caption': img_caption}
        results.append(result)
    return results


if __name__ == '__main__':
    beam_size = 10
    data_type = 'TRAIN'
    result_csv = f'./ckpts/benchmarks_{data_type.lower()}.csv'
    
    agg_results = []
    print(f'beam size: {beam_size}')
    results = run_test_per_beamsize_style(beam_size, data_type = data_type, n = 200)

    if agg_results == []:
        agg_results = results
    else:
        for i in range(len(agg_results)):
            agg_results[i].update(results[i])

    result_df = pd.DataFrame(agg_results)
    result_df.to_csv(result_csv, index = False)
    print(f'result csv written: {result_csv}')
