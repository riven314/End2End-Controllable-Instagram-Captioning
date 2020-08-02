#!/bin/bash
python prepare_input.py --img_dir ../data/Instagram/images --min_word_freq 1 --min_len 50 --json_path ./data/ig_json/full_clean_wonumber_wemojis.json --out_dir ./data/meta_wstyle/data_full_clean_wonumber_wemojis_wp --is_write_img
python main.py
python inference.py 
