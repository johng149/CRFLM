import torch

def custom_loss(logits, targets, crf_losses, vocab_size, pad_idx, nll_loss_weight, answer_len):
    bt_size, _ = targets.shape
    cross_entropy_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.view(-1), ignore_index=pad_idx, reduction='none').view(bt_size, answer_len)
    target_padding_matrix = ~targets.eq(pad_idx)
    target_padding_matrix = target_padding_matrix.type(cross_entropy_loss.type())
    cross_entropy_loss = cross_entropy_loss * target_padding_matrix
    summed_loss = cross_entropy_loss.sum(dim = -1)
    one_step_train_loss = crf_losses + nll_loss_weight * summed_loss
    scaled_train_loss = torch.sum(one_step_train_loss) / torch.sum(target_padding_matrix)
    return scaled_train_loss