import torch
from torch.nn import Module, LayerNorm
from torch.fft import fft
from torch import real

from .phm import phm

# https://nn.labml.ai/transformers/fnet/index.html
class FNet(Module):

    def __init__(self, embed_dim, phm_factor=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_proj = phm(phm_factor, embed_dim, embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, embeddings):
        result = fft(fft(self.norm1(embeddings), dim=2), dim=0)
        result = real(result) + embeddings
        result = self.out_proj(self.norm2(result)) + result
        return result