import torch

def nag_bert_loss(logits, crf_losses, answer, pad_idx, nll_loss_weight=0.5):
    batch_size, seq_len = answer.shape
    logits_unbind = torch.unbind(logits.transpose(0, 1), dim=0)
    answer_unbind = torch.unbind(answer, dim=1)
    nll = torch.nn.NLLLoss(ignore_index=pad_idx, reduction='none')
    nll_losses = []
    for i in range(seq_len):
        curr_logits = logits_unbind[i]
        curr_answer = answer_unbind[i].view(batch_size)
        curr_nll = nll(curr_logits, curr_answer)
        nll_losses.append(curr_nll)
    nll_losses = torch.stack(nll_losses, dim=1)
    answer_padding_matrix = ~answer.eq(pad_idx)
    answer_padding_matrix = answer_padding_matrix.type(nll_losses.type())
    nll_losses = nll_losses * answer_padding_matrix
    nll_losses_sum = nll_losses.sum(dim=-1)
    loss = nll_losses_sum*nll_loss_weight + crf_losses
    loss = torch.sum(loss) / torch.sum(answer_padding_matrix)
    return loss