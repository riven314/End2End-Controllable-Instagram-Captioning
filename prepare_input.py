import os
import argparse

from src.utils import create_input_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type = str, default = '/media/alex/Data/personal/Project/MadeWithML_Incubator/data/instagram/images')
    parser.add_argument('--min_word_freq', type = int, default = 5)
    parser.add_argument('--min_len', type = int, default = 50)
    parser.add_argument('--json_path', type = str, required = True)
    parser.add_argument('--out_dir', type = str, required = True)
    args = parser.parse_args()

    print(f'img_dir: {args.img_dir}')
    print(f'json_path: {args.json_path}')
    print(f'out_dir: {args.out_dir}')
    print(f'min_word_freq: {args.min_word_freq}')
    print(f'min_len: {args.min_len}')

    # Create input files (along with word map)
    os.makedirs(args.out_dir, exist_ok = True)
    create_input_files(dataset = 'flickr8k',
                       karpathy_json_path = args.json_path,
                       image_folder = args.img_dir,
                       min_word_freq = args.min_word_freq,
                       output_folder = args.out_dir,
                       max_len = args.min_len)