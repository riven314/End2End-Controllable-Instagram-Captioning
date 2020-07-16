import os

from cfg import Config
from models import get_encoder_decoder

cfg = Config()
cfg.checkpoint = None
encoder_wockpt, decoder_wockpt = get_encoder_decoder(cfg)
print('encoder, decoder without checkpoint loaded')

checkpoint_path = './ckpts/v1/BEST_checkpoint_flickr8k_1_cap_per_img_5_min_word_freq.pth'
if os.path.isfile(checkpoint_path):
    cfg.checkpoint = checkpoint_path
    encoder_wckpt, decoder_wckpt = get_encoder_decoder(cfg)
    print('encoder decoder without checkpoint loaded')
