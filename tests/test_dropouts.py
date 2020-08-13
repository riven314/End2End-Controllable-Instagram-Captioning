import os

import torch
import torch.nn as nn
import numpy as np

from src.dropouts import WeightDropout, InputDropout, RNNDropout, EmbeddingDropout


def test_WeightDropout():
    p = 0.8
    lstm = nn.LSTMCell(2, 4)
    dp_lstm = WeightDropout(lstm, weight_p = p, layer_names = ['weight_hh'])

    test_inp = torch.randn(8, 2)
    test_h, test_c = torch.randn(8,4), torch.randn(8, 4)

    assert dp_lstm.training is True
    assert dp_lstm.weight_hh_raw.requires_grad is True

    # check dropout mask is applied on target weight matrices with proper scaling
    weight_before = dp_lstm.module.weight_hh.data.clone()
    h, c = dp_lstm(test_inp, (test_h, test_c), reset_mask = True)
    weight_after_reset = dp_lstm.module.weight_hh.data.clone()
    assert torch.logical_or(weight_after_reset == 0, (weight_before/(1-p) - weight_after_reset).abs() < 1e-5).all()

    # check gradients are computed, dropout entries have grad = 0
    loss = h.sum()
    loss.backward()
    assert dp_lstm.weight_hh_raw.grad is not None 
    assert torch.logical_and(dp_lstm.weight_hh_raw.grad == 0, dp_lstm.module.weight_hh == 0).any()

    # check dropout mask is fixed when reset_mask = False
    dp_lstm.zero_grad()
    h, c = dp_lstm(test_inp, (test_h, test_c), reset_mask = False)
    weight_without_reset = dp_lstm.module.weight_hh.data.clone()
    assert (weight_without_reset == weight_after_reset).all()


def test_EmbeddingDropout():
    p = 0.8
    embedding = nn.Embedding(10, 3)
    test_inp = torch.LongTensor([0, 1, 4, 5, 6,7])
    dp_embedding = EmbeddingDropout(embedding, embed_p = p)
    out = dp_embedding(test_inp)

    # check dropout is applied on embedding layer
    assert dp_embedding.emb.weight.requires_grad is True
    assert (out.sum(axis = 1) == 0).any()


def test_RNNDropout():
    p = 0.8
    lstm = nn.LSTMCell(2, 4)
    test_inp = torch.randn(8, 2)
    test_h, test_c = torch.randn(8,4), torch.randn(8, 4)
    hiddden_dp = RNNDropout(p = p)

    out, _ = lstm(test_inp, (test_h, test_c))
    
    dp_out = hiddden_dp(out, reset_mask = True)
    dp_wreset_out = hiddden_dp(out, reset_mask = True)
    dp_woreset_out = hiddden_dp(out, reset_mask = False)
    assert not (dp_out == dp_wreset_out).all()
    assert (dp_wreset_out == dp_woreset_out).all()


def test_InputDropout():
    seq_len = 20
    inp_dp = InputDropout(p = 0.8)
    test_inp = torch.randn(8, seq_len, 10)
    out = inp_dp(test_inp)
    
    prev_mask = None
    for t in range(seq_len):
        prev_mask = (out[:, t, :] == 0) if prev_mask is None else prev_mask
        crt_mask = (out[:, t, :] == 0)
        assert (prev_mask == crt_mask).all()
        prev_mask = crt_mask