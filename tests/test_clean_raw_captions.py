import os
import json

import re

from data.clean_raw_captions import process_character_space


def test_process_character_space():
    test_pos = ['f u t u r e', 'b a t', 'c a v e !']
    test_neg = ['alex', 'its the best moment']

    print('\n')
    for sen in test_pos:
        new_sen, boo = process_character_space(sen)
        assert boo is True
        print(f'before: {sen} ==> after: {new_sen}')
    
    for sen in test_neg:
        new_sen, boo = process_character_space(sen)
        assert boo is False
        print(f'before: {sen} ==> after: {new_sen}')


if __name__ == '__main__':
    test_process_character_space()