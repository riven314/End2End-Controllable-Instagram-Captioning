""" create json file as input for create_input_files.py """
import os
import json
from shutil import copyfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from data.utils import read_txt, write_json, decode_one_sample


ROOT = '/media/alex/Data/personal/Project/MadeWithML_Incubator/data/instagram'
DATA_DIR = os.path.join(ROOT, 'caption_dataset')
IMG_DIR = os.path.join(ROOT, 'images')
VOCAB_PATH = os.path.join(DATA_DIR, '40000.vocab')
OUT_DIR = os.path.join(os.getcwd(), 'data', 'ig_json')
os.makedirs(OUT_DIR, exist_ok = True)

vocab = read_txt(VOCAB_PATH)


def get_input_json(txt_ls, vocab, data_split, json_path):
    outs = []
    for txt_data_path, data_type in zip(txt_ls, data_split):
        txt_data = read_txt(txt_data_path)

        for i, sample in tqdm(enumerate(txt_data)):
            fn, token_seq, word_seq = decode_one_sample(sample, vocab)
            out = {'sentences': [{'tokens': word_seq}], 
                   'split': data_type,
                   'filename': fn}
            outs.append(out)

    outs = {'images': outs}
    write_json(outs, json_path)
    print(f'json written: {json_path}')

    return None


if __name__ == '__main__':
    tup = [('trial.txt', 'train'),
           ('trial.txt', 'val'),
           ('trial.txt', 'test')]
    json_path = os.path.join(OUT_DIR, 'trial.json')
    
    tup = [(os.path.join(DATA_DIR, txt_fn), data_type) for txt_fn, data_type in tup]
    txt_ls, data_split = zip(*tup)

    get_input_json(txt_ls, vocab, data_split, json_path)