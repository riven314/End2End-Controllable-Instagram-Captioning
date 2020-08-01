import os

from data.utils import read_json
from src.word_map_utils import create_word_map_from_simple
from src.word_map_utils import create_word_map_from_pretrained_wordpiece

ig_json = './data/ig_json/full_clean_wonumber.json'
output_folder = './tests'
data = read_json(ig_json)


def test_create_word_map_from_pretrained_wordpiece():
    word_map, all_captions = create_word_map_from_pretrained_wordpiece(
        data, 'wordpiece', output_folder = output_folder, 
        min_word_freq = 5, max_len = 50, vocab_size = 8000
    )
    return word_map, all_captions


def test_create_word_map_from_simple():
    word_map, all_captions = create_word_map_from_simple(
        data, 'simple', output_folder = output_folder,
        min_word_freq = 5, max_len = 50, vocab_size = None
    )
    return word_map, all_captions


if __name__ == '__main__':
    simple_word_map, simple_all_captions = test_create_word_map_from_simple()
    wp_word_map, wp_all_captions = test_create_word_map_from_pretrained_wordpiece()
