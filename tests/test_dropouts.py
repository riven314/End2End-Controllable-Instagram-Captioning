import torch
import torch.nn as nn

from src.dropouts import WeightDropout

# test WeightDropout
# the target masked weight no longer has required_grads
# self.weight_raw will be updated after self.module.weight is updated?
p = 0.8
module = nn.LSTMCell(2, 4)
lstm = WeightDropout(module, weight_p = p, layer_names = ['weight_hh'])

test_input = torch.randn(8, 2)
test_h = torch.randn(8, 4)
test_c = test_h.data
weight_before = lstm.module.weight_hh.data

h, c = lstm(test_input, (test_h, test_c), reset_mask = True)
weight_after_reset = lstm.module.weight_hh.data
assert ((weight_after_reset == 0) + (weight_after_reset == weight_before / (1 - p))).all()

h, c = lstm(test_input, (test_h, test_c), reset_mask = False)
weight_after_retain = lstm.module.weight_hh.data
assert (weight_after_reset == weight_after_retain).all()
