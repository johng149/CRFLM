import torch
from torch.nn import Module
from .embedding import StableEmbedding

class Reader(Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = StableEmbedding(vocab_size, embed_dim)

    def forward(self, x):
        result = self.embedding(x)
        # at this point result is batch_size x seq_len x embed_dim
        # however, the cosformer attention expects
        # seq_len x batch_size x embed_dim so we transpose
        return result.transpose(0, 1)