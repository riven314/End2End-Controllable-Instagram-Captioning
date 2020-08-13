import os
import json

from easydict import EasyDict as edict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from src.decoder import DecoderWithAttention, RegularizedDecoderWithAttention


def get_encoder_decoder(cfg):
    if isinstance(cfg, dict):
        cfg = edict(cfg)

    with open(cfg.word_map_file, 'r') as f:
        word_map = json.load(f)

    encoder = Encoder()
    encoder.fine_tune(cfg.fine_tune_encoder)

    if cfg.regularized_decoder is None:
        decoder = DecoderWithAttention(attention_dim = cfg.attention_dim,
                                    embed_dim = cfg.emb_dim,
                                    decoder_dim = cfg.decoder_dim,
                                    vocab_size = len(word_map),
                                    dropout = cfg.dropout)
    else:
        dropout_dict = cfg.regularized_decoder
        fc_p, embed_p  = dropout_dict['fc_p'], dropout_dict['embed_p']
        weight_p, input_p = dropout_dict['weight_p'], dropout_dict['input_p']
        decoder = RegularizedDecoderWithAttention(attention_dim = cfg.attention_dim,
                                                  embed_dim = cfg.emb_dim,
                                                  decoder_dim = cfg.decoder_dim,
                                                  vocab_size = len(word_map),
                                                  fc_p = fc_p, embed_p = embed_p,
                                                  weight_p = weight_p, input_p = input_p)

    if cfg.checkpoint_file is not None:
        assert os.path.isfile(cfg.checkpoint_file)
        print(f'load in checkpoint: {cfg.checkpoint_file}')
        checkpoint = torch.load(cfg.checkpoint_file)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
    return encoder, decoder


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


