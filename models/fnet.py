import torch
from torch.nn import Module
from torch.fft import fft
from torch import real

from .phm import phm

# https://nn.labml.ai/transformers/fnet/index.html
class FNet(Module):

    def __init__(self, embed_dim=None, phm_factor=4):
        self.embed_dim = embed_dim
        if embed_dim is not None:
            # then after the fft we have feedforward
            self.out_proj = phm(phm_factor, embed_dim, embed_dim)

    def forward(self, embeddings):
        result = fft(fft(embeddings, dim=2), dim=0)
        if self.embed_dim is not None:
            result = self.out_proj(result)
        return real(result)