from src.datasets import get_dataloaders
from config.cfg import Config

cfg = Config()
train_loader, val_loader, test_loader = get_dataloaders(cfg)