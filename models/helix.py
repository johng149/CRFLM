# I call it Helix because it resembles a DNA helix
import torch
from torch.nn import Module, LayerNorm
from torch.nn import AdaptiveAvgPool1d

from .fnet import FNet
from .cosformer import CosformerAttention as Attention

class Helix(Module):

    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()

        # layers for query tokens
        self.f1q = FNet(embed_dim, phm_factor)
        self.selfq = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.f2q = FNet(embed_dim, phm_factor)

        # layers for context tokens
        self.f1c = FNet(embed_dim, phm_factor)
        self.selfc = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.f2c = FNet(embed_dim, phm_factor)

        # layer where the query and context tokens interact
        self.context_kv_to_q = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.query_kv_to_c = Attention(embed_dim, num_heads, phm_factor=phm_factor)

    def _process_query(self, query):
        # input is seq_len x batch_size x embed_dim
        result = self.f1q(query)
        result = self.selfq(result)
        result = self.f2q(result)
        return result
    
    def _process_context(self, context):
        # input is seq_len x batch_size x embed_dim
        result = self.f1c(context)
        result = self.selfc(result)
        result = self.f2c(result)
        return result
    
    def _process_mixing(self, query, context):
        # input is seq_len x batch_size x embed_dim
        # output is seq_len x batch_size x embed_dim
        context_with_query_kv = self.context_kv_to_q(context, query, query)
        query_with_context_kv = self.query_kv_to_c(query, context, context)
        return query_with_context_kv, context_with_query_kv
    
    def forward(self, query, context):
        # input is seq_len x batch_size x embed_dim
        # output is seq_len x batch_size x embed_dim
        query = self._process_query(query)
        context = self._process_context(context)
        query, context = self._process_mixing(query, context)
        return query, context
    
class Endcap(Module):

    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()

        self.query_kv_to_c = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.f1 = FNet(embed_dim, phm_factor)
        self.selfc = Attention(embed_dim, num_heads, phm_factor=phm_factor)

    def forward(self, query, context):
        context_with_query_kv = self.query_kv_to_c(context, query, query)
        result = self.f1(context_with_query_kv)
        result = self.selfc(result)
        return result
    
class Downsizer(Module):

    def __init__(self, embed_dim, num_heads, target_size, phm_factor=4):
        super().__init__()

        self.pool = AdaptiveAvgPool1d(target_size)
        self.pooled_with_x_kv = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.f = FNet(embed_dim, phm_factor)

    def forward(self, x):
        # assumes that x is of shape seq_len, batch_size, embed_dim
        # and that seq_len >= target_size
        pooled = self.pool(x.transpose(0, 2))

        # now pooled is of shape embed_dim, batch_size, target_size
        # however, we want target_size, batch_size, embed_dim
        pooled = pooled.transpose(0, 2)
        pooled = self.pooled_with_x_kv(pooled, x, x)
        return self.f(pooled)
    
class SingleStrand(Module):

    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()

        # layers for query tokens
        self.f1 = FNet(embed_dim, phm_factor)
        self.attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.f2 = FNet(embed_dim, phm_factor)

    def forward(self, x):
        # input is seq_len x batch_size x embed_dim
        result = self.f1(x)
        result = self.attn(result)
        result = self.f2(result)
        return result