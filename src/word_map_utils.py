import os
import json
from collections import Counter

import emoji
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer

from data.utils import write_json
from data.clean_raw_captions import extract_emoji


def get_wp_tokenizer(data):
    print('collecting all emojis of the data...')
    emoji_set = set()
    for id_dict in tqdm(data['images']):
        caption = ' '.join(id_dict['sentences'][0]['tokens'])
        emojis = extract_emoji(caption)
        if emojis is None:
            continue
        emoji_set.update(emojis)
    emoji_set = list([emoji.demojize(e) for e in emoji_set])
    print(f'total no. of emojis: {len(emoji_set)}')

    # setup tokenizer and add special tokens 
    wp_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    wp_tokenizer.add_tokens('@username')
    wp_tokenizer.add_tokens(emoji_set)
    return wp_tokenizer


def create_word_map_from_simple(data, base_filename, output_folder, min_word_freq, max_len, vocab_size = None):
    # create and save word_map
    word_freq = Counter()
    all_captions = []

    print('building word_freq...')
    for img in tqdm(data['images']):
        tokens = img['sentences'][0]['tokens'].copy()

        captions = []
        if len(tokens) <= max_len:
            captions.append(tokens)
            word_freq.update(tokens)
        all_captions.append(captions)

    rare_word_cnt = sum([cnt for w, cnt in word_freq.items() if cnt <= min_word_freq])
    total_cnt = sum(word_freq.values())
    print(f'proportion of invalid tokens: {rare_word_cnt/total_cnt}')

    # more than min_word_freq
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    json_path = os.path.join(output_folder, f'WORDMAP_{base_filename}.json')
    write_json(word_map, json_path)

    return word_map, all_captions


def create_word_map_from_pretrained_wordpiece(data, base_filename, output_folder, min_word_freq, max_len, vocab_size = None):
    """
    :return word_map: subword vocab dict
    :all_captions: list of list (each are list of subwords)
    """
    wp_tokenizer = get_wp_tokenizer(data)

    subword_freq = Counter()
    all_captions = []

    print('building subword_freq...')
    for img in tqdm(data['images']):
        tokens = img['sentences'][0]['tokens'].copy()

        captions = []
        if len(tokens) <= max_len:
            caption = ' '.join(tokens)
            caption = emoji.demojize(caption)
            wp_tokens = wp_tokenizer.tokenize(caption)
            subword_freq.update(wp_tokens)
            captions.append(wp_tokens)
        all_captions.append(captions)
    
    # clean out [UNK] token, also count proportion of invalid tokens
    if '[UNK]' in subword_freq:
        unk_cnt = subword_freq['[UNK]']
        del subword_freq['[UNK]']
    rare_word_cnt = sum([cnt for w, cnt in subword_freq.items() if cnt <= min_word_freq])
    total_cnt = sum(subword_freq.values())
    print(f'proportion of invalid tokens: {rare_word_cnt/total_cnt}')
    
    # build sub word map
    words = [w for w in subword_freq.keys() if subword_freq[w] > min_word_freq]

    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    json_path = os.path.join(output_folder, f'WORDMAP_{base_filename}.json')
    write_json(word_map, json_path)

    return word_map, all_captions


