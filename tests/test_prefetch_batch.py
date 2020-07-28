import os

import numpy as np
from tqdm import tqdm
import torch

from prefetch_generator import BackgroundGenerator

from src.models import get_encoder_decoder
from src.datasets import get_dataloaders
from config.cfg import Config


cfg = Config()
cfg.data_folder = './data/meta_wstyle/data_mid_clean_wonumber'
#cfg.data_folder = '/home/alex/Desktop/data_mid_clean_wonumber'
cfg.data_name = 'flickr8k_1_cap_per_img_1_min_word_freq'
cfg.workers = 3
cfg.batch_size = 64

device = torch.device('cuda')

encoder, decoder = get_encoder_decoder(cfg)
encoder.to(device)
decoder.to(device)

train_dl, val_dl, test_dl = get_dataloaders(cfg)

for i, (imgs, caps, caplens, _, len_class, img_ids) in enumerate(BackgroundGenerator(train_dl), 0):
    if i == 100:
        break
    imgs = imgs.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)
    len_class = len_class.to(device)

    enc_out = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(enc_out, caps, caplens, len_class)