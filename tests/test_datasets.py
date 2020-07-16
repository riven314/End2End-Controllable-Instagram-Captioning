import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import *


data_folder = '../data/meta_wstyle/data_trial'
data_name = 'flickr8k_1_cap_per_img_5_min_word_freq'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
train_ds = CaptionDataset(data_folder, data_name, 'TRAIN', 
                          transform=transforms.Compose([normalize]))
val_ds = CaptionDataset(data_folder, data_name, 'VAL', 
                        transform=transforms.Compose([normalize]))

train_dl = DataLoader(train_ds, batch_size = 8, shuffle = True, 
                      num_workers = 1, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size = 8, shuffle = False, 
                    num_workers = 1, pin_memory = True)

# QQQ@Alex: are caption length sorted in dataloader in batch order?
train_sample_dict = dict()
val_sample_dict = dict()

for i, (train_sample, val_sample) in enumerate(zip(train_dl, val_dl)):
    if i == 6:
        break
    train_sample_dict[i] = train_sample
    val_sample_dict[i] = val_sample
    
    assert len(train_sample) == 6
    assert len(val_sample) == 6
    