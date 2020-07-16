import torch

class Config:
    save_dir = './ckpts/v2'
    data_folder = 'data/meta_wstyle/data_mid'
    data_name = 'flickr8k_1_cap_per_img_5_min_word_freq'
    checkpoint_file = './ckpts/v1/BEST_checkpoint_flickr8k_1_cap_per_img_5_min_word_freq.pth'
    word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'
    
    attention_dim = 512
    emb_dim = 512
    decoder_dim = 512
    dropout = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 120
    workers = 1
    batch_size = 64

    fine_tune_encoder = False
    encoder_lr = 1e-4
    decoder_lr = 4e-3 # 4e-4
    grad_clip = 5.
    alpha_c = 1.

    best_bleu4 = 0.
    print_freq = 50
    checkpoint = None
    tolerance_epoch = 8
    adjust_epoch = 2
    adjust_step = 0.6

    
    

