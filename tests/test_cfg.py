import os
import json

from cfg import Config


save_path = './tests/test.json'

cfg = Config()
cfg.save_dir = 'ALEX TESTING'
cfg.batch_size = 144

cfg.save_config(save_path)

with open(save_path, 'r') as f:
    saved_dict = json.load(f)


assert saved_dict['save_dir'] == 'ALEX TESTING'
assert saved_dict['batch_size'] == 144