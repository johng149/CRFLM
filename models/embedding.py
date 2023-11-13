import torch.nn as nn

class StableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(StableEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)
