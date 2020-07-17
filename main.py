import os

import torch.backends.cudnn as cudnn

from cfg import Config
from models import get_encoder_decoder
from datasets import get_dataloaders
from learner import Learner

cudnn.benchmark = True
cfg = Config()

encoder, decoder = get_encoder_decoder(cfg)
train_loader, val_loader, test_loader = get_dataloaders(cfg)


if __name__ == '__main__':
    test_loader = None
    learner = Learner(encoder, decoder, train_loader, val_loader, test_loader, cfg)
    learner.main_run()
