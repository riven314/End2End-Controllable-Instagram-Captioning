import os
import re
import json
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd

import emoji
import spacy
from spacy_hunspell import spaCyHunSpell
from transformers import pipeline


def basic_clean_one_caption(caption):
    """
    :return new_caption:
    :return is_emoji:
    :return is_hashtag:
    """
    # lower case and strip white-space on two ends
    new_caption = caption.lower().strip()
    # handle character-space pattern (e.g. b a t)
    new_caption, is_concat = process_character_space(new_caption)

    # remove duplicate consecutive punctuation

    return new_caption


def process_character_space(text):
    """
    :return text: cleaned text (either unchanged/ concatenated by character)
    :return bool: whether text is procesesd by this func
    """
    tokens = text.split()
    tokens_len = [len(token) for token in tokens]
    if all([_len == 1 for _len in tokens_len]) and len(tokens) >= 3:
        return (text.replace(' ', ''), True)
    else:
        return (text, False)


def demojize(text):
    """ change emoji to a code e.g. :hugging_face: """
    return emoji.demojize(text)



def replace_person_name(text):
    """ assume person name is correctly spelled """
    pass


def replace_username(text):
    """ e.g. "@riven_hong.asdas-314 hi" -> "@username hi" """
    return re.sub(r'@[\S]+', '@username', text)


def get_hashtag_positions(text):
    """
    e.g. "#tbt kind! #summer" -> [0, 11]
    """
    positions = [pos for pos, char in enumerate(text) if char == '#']
    if len(positions) == 0:
        return None
    else:
        return positions


def space_out_some_characters(text):
    """ e.g. "its the best@username" -> "its the best @username" """
    pass



#re.compile(r'([/#\\])')

# replace repetition at character level
# _re_rep = re.compile(r'(\S)(\1{2,})')
# def replace_rep(t):
#     "Replace repetitions at the character level: cccc -- TK_REP 4 c"
#     def _replace_rep(m):
#         c,cc = m.groups()
#         return f' {TK_REP} {len(cc)+1} {c} '
#     return _re_rep.sub(_replace_rep, t)