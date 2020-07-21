import os
import json
from collections import Counter

import numpy as np
from utils import read_json, write_json


def remove_backslash(json_data):
    new_json_data = []
    for id_dict in json_data['images']:
        tokens = id_dict['sentences'][0]['tokens']
        tokens = [t for t in tokens if t != '\\']
        id_dict['sentences'][0]['tokens'] = tokens
        new_json_data.append(id_dict)
    new_json_data = {'images': new_json_data}
    return new_json_data


def remove_unk(json_data):
    new_json_data = []
    for id_dict in json_data['images']:
        tokens = id_dict['sentences'][0]['tokens']
        if '_UNK' in tokens:
            continue
        new_json_data.append(id_dict)
    new_json_data = {'images': new_json_data}
    return new_json_data


def truncate_ngram(json_data, n, cap = 300):
    ngram_counter = Counter()

    new_json_data = []
    for id_dict in json_data['images']:
        tokens = id_dict['sentences'][0]['tokens']
        tokens_ngram = tuple(tokens[:n])
        
        if tokens_ngram not in ngram_counter:
            ngram_counter[tokens_ngram] = 1
        if ngram_counter[tokens_ngram] > cap:
            continue
            
        new_json_data.append(id_dict)
        ngram_counter.update([tokens_ngram])
    new_json_data = {'images': new_json_data}
    return new_json_data
    

def create_styles(json_data):
    new_json_data = []
    for id_dict in json_data['images']:
        tokens = id_dict['sentences'][0]['tokens']
        sentence_length = len(tokens)
        id_dict['styles'] = {'sentence_length': sentence_length}
        new_json_data.append(id_dict)
    new_json_data = {'images': new_json_data}
    return new_json_data


if __name__ == '__main__':
    JSON_PATH = './ig_json/full.json'
    json_data = read_json(JSON_PATH)
    
    new_json_data = remove_backslash(json_data)
    new_json_data = remove_unk(new_json_data)
    new_json_data = truncate_ngram(new_json_data, n = 3, cap = 400)
    new_json_data = truncate_ngram(new_json_data, n = 2, cap = 700)
    new_json_data = truncate_ngram(new_json_data, n = 1, cap = 4000)
    new_json_data = create_styles(new_json_data)
    
    write_json(new_json_data, './ig_json/full_clean.json')