import os

import torch.backends.cudnn as cudnn

from config.cfg import Config
from src.models import get_encoder_decoder
from src.datasets import get_dataloaders
from src.learner import Learner

cudnn.benchmark = True

cfg = Config()
encoder, decoder = get_encoder_decoder(cfg)
train_loader, val_loader, test_loader = get_dataloaders(cfg)


if __name__ == '__main__':
    test_loader = None
    
    os.makedirs(cfg.save_dir, exist_ok = True)
    save_cfg_path = os.path.join(cfg.save_dir, 'config.json')
    cfg.save_config(save_cfg_path)

    learner = Learner(encoder, decoder, train_loader, val_loader, test_loader, cfg)
    learner.main_run()
