import torch


class Config:
    data_folder = 'data/meta_wstyle/data_mid'
    data_name = 'flickr8k_1_cap_per_img_5_min_word_freq'
    checkpoint_file = './ckpts/v1/BEST_checkpoint_flickr8k_1_cap_per_img_5_min_word_freq.pth'
    word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'
    checkpoint = None

    attention_dim = 512
    emb_dim = 512
    decoder_dim = 512
    dropout = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

