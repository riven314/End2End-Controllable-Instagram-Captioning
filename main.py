import os
import argparse

import torch.backends.cudnn as cudnn

from config.cfg import Config
from src.models import get_encoder_decoder
from src.datasets import get_dataloaders
from src.learner import Learner

cudnn.benchmark = True
cfg = Config()

encoder, decoder = get_encoder_decoder(cfg)
train_loader, val_loader, test_loader = get_dataloaders(cfg)
test_loader = None


if __name__ == '__main__':
    # cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = cfg.save_dir)
    parser.add_argument('--confidence_c', type = float, default = cfg.confidence_c)
    args = parser.parse_args()
    cfg.save_dir = args.save_dir
    cfg.confidence_c = args.confidence_c
    
    # init save_dir and save config.json
    os.makedirs(cfg.save_dir, exist_ok = True)
    save_cfg_path = os.path.join(cfg.save_dir, 'config.json')
    cfg.save_config(save_cfg_path)

    # init and start training
    learner = Learner(encoder, decoder, train_loader, val_loader, test_loader, cfg)
    learner.main_run()
