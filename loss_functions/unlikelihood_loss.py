import torch
from torch.functional import F
from math import ceil

def p_solver(o, k, s=1, d=1):
    """
    Solves for `p` such that for the given kernel size `k`, stride `s`, and dilation `d`,
    and desired output size `o`, a Conv2d layer with the given `p` will produce an output
    such that the shape of the input is preserved.
    """
    p = (d*k - d + o*s - o -s + 1)/2
    return ceil(p)

def diag_mask(x, k):
    """
    x is a 3D tensor where its last two dimensions are the same size.

    Will create a 2D matrix where the elements adjacent to the diagonal are 1, and the
    rest are 0, including the diagonal itself

    What qualifies as adjacent depends on `k`. For example,
    `k` = 1 means only the diagonal (but since diagonal elements are 0, will create all 0s matrix)
    `k` = 2 means the diagonal and the immediate adjacent elements
    and so on.

    `k` must be less than or equal to the size of the last two dimensions of `x`
    """
    kernel = torch.ones(k, k)
    n = x.shape[-1]
    diagonal = torch.eye(n)
    p = p_solver(n, k)
    result = F.conv2d(diagonal.view(1,1,n,n), kernel.view(1,1,k,k), padding=p)
    result = result.squeeze(0).squeeze(0)
    return (result[:n, :n].fill_diagonal_(0)).bool().int()

def overcounted_zeros(mask):
    """
    Given mask, determine the number of overcounted zeros per row

    This is done by summing along the final dimension and then subtracting the size
    scalar by the resulting 1d tensor, we get the number of 0s in each row,
    which is the number of overcounted zeros per row.
    """
    size = mask.shape[-1]
    return size - torch.sum(mask, dim=-1)

def unlikelihood_loss(logits, targets, k, allow_self_repeats_idx=None):
    """
    Unlikelihood loss penalizes the model for predicting high probability
    for tokens that are the same as nearby tokens.

    For example, if the targets are [0, 1, 2] and k = 2, then there will
    be penalities for predicting high probability of token 1 at position 0,
    and predicting high probability of token 0 or 2 at position 1, and
    predicting high probability of token 1 at position 2.

    If there are multiple tokens such as [0, 1, 0] and k = 2, then there will
    penalty for predictin high probability of token 0 at position 1, and this
    penalty will be doubled.

    `k` = 1 means only the diagonal is considered, but since we don't want to
    penalize the token itself, this is basically a no-op.

    `k` = 2 means the diagonal and the immediate adjacent elements are considered.

    if `allow_self_repeats_idx` is not None, then for the specified token, it is
    allowed to be repeated without penalty. This is used for things like padding
    tokens which are often repeated
    """
    with torch.no_grad():
        _, seq_len, _ = logits.shape
        if allow_self_repeats_idx is not None:
            is_allowed = targets == allow_self_repeats_idx
        indices_targets = targets.unsqueeze(1)
        indices_targets = indices_targets.expand(-1, seq_len, seq_len)
        mask = diag_mask(indices_targets, k)
        mask = mask.to(logits.device)
        indices_mask = torch.zeros_like(logits).scatter_(2, indices_targets*mask, 1, reduce='add')
        overcounted = overcounted_zeros(mask)
        indices_mask[:, :, 0] = indices_mask[:, :, 0] - overcounted
        if allow_self_repeats_idx is not None:
            indices_mask[:, :, allow_self_repeats_idx] = indices_mask[:, :, allow_self_repeats_idx] * ~is_allowed
            # you might be wondering "Why ~is_allowed instead of is_allowed? Why take the not?"
            # well the reason is, by doing this, any rows that was not derived from the 
            # allowed_self_repeats_idx token will be multiplied with 1 (keeping the penalty)
            # while the rows that was derived from the allowed_self_repeats_idx token will be
            # multiplied with 0 (removing the penalty)
    probs = torch.log_softmax(logits, dim=-1)
    values = torch.clamp(1-torch.exp(probs), min=1e-5)
    ul_losses = -torch.log(values) * indices_mask
    return ul_losses.sum()