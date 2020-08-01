import os
import json

import emoji

from data.utils import write_json


def create_nonimage_input(data, word_map, all_captions, base_filename, image_folder, output_folder):
    """
    write id, styles data
    streamline (captions, id, styles, image path) by data partition (train/ val/ test)

    :param all_captions: list of list of caption (caption = list of tokens)
    :param image_folder: dir for reading image
    :param output_folder: dir for writing json input data (for training)
    :return partition_dict: dict[partition] : (image_paths, image_captions)
    """
    print('building non-image input...')

    # read image paths and captions for each image
    train_tup, val_tup, test_tup = [], [], []

    for img, captions in zip(data['images'], all_captions):
        if len(captions) == 0:
            continue

        # extract image path
        path = os.path.join(image_folder, img['filename'])

        # extract styles
        sentence_length = len(img['sentences'][0]['tokens'])
        if sentence_length <= 5:
            length_class = 0
        elif (sentence_length > 5) & (sentence_length <= 9):
            length_class = 1
        else:
            length_class = 2

        is_emoji = any([token.startswith(':') and token.endswith(':') for token in captions[0]]) * 1
        style_dict = {'length_class': length_class, 'is_emoji': is_emoji}

        # streamline by data partition
        if img['split'] == 'train': # this is slow
            train_tup.append((img['filename'], path, captions, style_dict))
            pass
        elif img['split'] == 'val':
            val_tup.append((img['filename'], path, captions, style_dict))
        elif img['split'] == 'test':
            test_tup.append((img['filename'], path, captions, style_dict))

    
    # sanity check
    train_id, train_image_paths, train_image_captions, train_styles = zip(*train_tup)
    val_id, val_image_paths, val_image_captions, val_styles = zip(*val_tup)
    test_id, test_image_paths, test_image_captions, test_styles = zip(*test_tup)

    assert len(train_id) == len(train_image_paths) == len(train_image_captions) == len(train_styles)
    assert len(val_id) == len(val_image_paths) == len(val_image_captions) == len(val_styles)
    assert len(test_id) == len(test_image_paths) == len(test_image_captions) == len(test_styles)

    # save id and styles data for train/ val/ test
    for partition in ['TRAIN', 'VAL', 'TEST']:
        id_json_path = os.path.join(output_folder, f'{partition}_ID_{base_filename}.json')
        id_data = locals()[f'{partition.lower()}_id'] # fetch train_id/ val_id/ test_id
        write_json(id_data, id_json_path)

        style_json_path = os.path.join(output_folder, f'{partition}_STYLES_{base_filename}.json')
        styles_data = locals()[f'{partition.lower()}_styles'] # train_styles/ val_styles/ test_styles
        write_json(styles_data, style_json_path)

    partition_dict = dict(
        train = (train_image_paths, train_image_captions),
        val = (val_image_paths, val_image_captions),
        test = (test_image_paths, test_image_captions)
    )

    print('complete non-image input...')
    return partition_dict