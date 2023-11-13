# https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
# modified it a little to use the PHM linear layers

import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn

from .phm import phm
from torch.nn import LayerNorm

class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout_rate=0.0,
        act_fun="relu",
        phm_factor=4 # reasonable and probably safe default
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self.num_heads = num_heads
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.k_proj = phm(phm_factor, self.kdim, embed_dim)
        self.v_proj = phm(phm_factor, self.vdim, embed_dim)
        self.q_proj = phm(phm_factor, embed_dim, embed_dim)
        # outprojection
        self.out_proj = phm(phm_factor, embed_dim, embed_dim)
        # dropout rate
        self.dropout_rate = dropout_rate

        # layer norm before attention
        self.norm1 = LayerNorm(embed_dim)

        # layer norm after attention
        self.norm2 = LayerNorm(embed_dim)

        # removed causal since not needed for this task

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
        """
        if key == None:
            key = query
        else:
            key = self.norm1(key)
        if value == None:
            value = query
        else:
            value = self.norm1(value)
        query = self.norm1(query)
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        attn_output += query
        attn_output = self.norm2(attn_output)
        attn_output = self.out_proj(attn_output) + attn_output

        return attn_output

# def test(batch=2, tgt_len=10, src_len=20, embed_dim=128, num_heads=8, N=100, causal=False):
#     model = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal)
#     diff = 0
#     if causal:
#         mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf'))
#     else:
#         mask = None
#     for i in range(N):
#         query = torch.rand(tgt_len, batch, embed_dim)
#         key = torch.rand(src_len, batch, embed_dim)
#         value = torch.rand(src_len, batch, embed_dim)
#         left_res = model.left_product(query, key, value, mask)
#         right_res = model(query, key, value)
#         diff += torch.norm(left_res - right_res)
#     diff /= N

#     if causal:
#         print("Test result for causal model:")
#     else:
#         print("Test result for bidirectional model:")
#     print(f"The error of left multiplication and right multiplication is {diff}")

# def main():
#     test(tgt_len=10, src_len=20, causal=False)
#     test(tgt_len=10, src_len=10, causal=True)

# if __name__ == "__main__":
#     main()