import torch
from torch.nn import Module
from torch.fft import fft
from torch import real

# https://nn.labml.ai/transformers/fnet/index.html
class FNet(Module):

    def forward(self, embeddings):
        result = fft(fft(embeddings, dim=2), dim=0)
        return real(result)