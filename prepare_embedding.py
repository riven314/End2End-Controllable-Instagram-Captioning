""" prepare pretrained word embedding weight using pymagnitude """
import os
import json

import numpy as np
from tqdm import tqdm
from pymagnitude import *

from data.utils import read_json

MAGNITUDE = '/home/alex/.magnitude/glove.twitter.27B.200d.magnitude'


def build_embedding_weight(word_map, embed_dim = 200):
    vocab_size = len(word_map)
    weight = np.zeros((vocab_size, embed_dim))

    vectors = Magnitude(MAGNITUDE)
    for word, idx in tqdm(word_map.items()):
        word_embed = vectors.query(word)
        weight[idx, :] = word_embed
    return weight


if __name__ == '__main__':
    word_map = './data/meta_wstyle/data_mid_clean_wonumber/WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json'
    npy_file = './pretrained/embedding.npy'

    word_map = read_json(word_map)
    weight = build_embedding_weight(word_map)
    with open(npy_file, 'wb') as f:
        np.save(f, weight)
    print(f'pretrained word embedding written: {npy_file} ({weight.shape})')

    # rev_word_map = {v: k for k, v in word_map.items()}
    # test_word_map = dict()
    # for i in range(50):
    #     word = rev_word_map[i]
    #     idx = word_map[word]
    #     test_word_map[word] = idx



