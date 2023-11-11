# https://github.com/yxuansu/NAG-BERT/blob/main/dynamic_crf_layer.py
# taken straight from the above to figure out if it is an issue
# with my implementation of the CRF layer or something else
# seems most of my code doesn't work in context of this task
# so I added some comments to describe why the code is written the way it is

import torch
import torch.nn as nn
def logsumexp(x, dim=1):
    return torch.logsumexp(x.float(), dim=dim).type_as(x)

class CRF(nn.Module):
    def __init__(self, num_embedding, beam_size=64, low_rank=32):
        super().__init__()

        self.E1 = nn.Embedding(num_embedding, low_rank)
        self.E2 = nn.Embedding(num_embedding, low_rank)

        self.vocb = num_embedding
        self.rank = low_rank
        self.beam = beam_size

    def forward(self, emissions, targets, masks, beam=None):
        numerator = self._compute_score(emissions, targets, masks)
        denominator = self._compute_normalizer(emissions, targets, masks, beam)
        return numerator - denominator

    def _compute_score(self, logits, targets, masks=None):
        logit_scores = logits.gather(2, targets.unsqueeze(2))[:, :, 0]
        transition_scores = self.E1(targets[:, :-1]) * self.E2(targets[:, 1:])
        logit_scores[:, 1:] += transition_scores.sum(2)
        if masks is not None:
            logit_scores = logit_scores * masks.type_as(logit_scores)
        return logit_scores.sum(-1)

    def _compute_normalizer(self, emissions, targets=None, masks=None, beam=None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        if targets is not None:
            _emissions = emissions.scatter(2, targets[:, :, None], float('inf'))
            beam_targets = _emissions.topk(beam, 2)[1]
            beam_emission_scores = emissions.gather(2, beam_targets)
        else:
            beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D; position i - 1, previous step.
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D; position i, current step.

        # i thought that we could just do `.view(-1, beam, self.rank) @ ...view(-1, self.rank, beam)`
        # and the model would just learn appropriately, but apparently not,
        # doing so causes numerator - denominator to be too big
        beam_transition_matrix = beam_transition_score1.view(-1, beam, self.rank) @ beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2)
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        # compute the normalizer in the log-space
        # i see, it is possible to do logsumexp after score + transition + next_emission
        # but doing so causes the result to be in the wrong shape for masks
        # could get it to work with a second logsumexp, but that rather defeats the purpose
        # of doing logsumexp all at once
        score = beam_emission_scores[:, 0]  # B x K
        for i in range(1, seq_len):
            next_score = score[:, :, None] + beam_transition_matrix[:, i-1]
            next_score = logsumexp(next_score, dim=1) + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i:i+1], next_score, score)
            else:
                score = next_score

        # Sum (log-sum-exp) over all possible tags
        return logsumexp(score, dim=1)

    def viterbi_decode(self, emissions, masks=None, beam=None):
        beam = beam if beam is not None else self.beam
        batch_size, seq_len = emissions.size()[:2]
        beam_emission_scores, beam_targets = emissions.topk(beam, 2)
        beam_transition_score1 = self.E1(beam_targets[:, :-1])  # B x (T-1) x K x D
        beam_transition_score2 = self.E2(beam_targets[:, 1:])   # B x (T-1) x K x D
        beam_transition_matrix = beam_transition_score1.view(-1, beam, self.rank) @ beam_transition_score2.view(-1, beam, self.rank).transpose(1, 2)
        beam_transition_matrix = beam_transition_matrix.view(batch_size, -1, beam, beam)

        traj_tokens, traj_scores = [], []
        finalized_tokens, finalized_scores = [], []

        # compute the normalizer in the log-space
        score = beam_emission_scores[:, 0]  # B x K
        dummy = torch.arange(beam, device=score.device).expand(*score.size()).contiguous()

        for i in range(1, seq_len):
            traj_scores.append(score)
            _score = score[:, :, None] + beam_transition_matrix[:, i-1]
            _score, _index = _score.max(dim=1)
            _score = _score + beam_emission_scores[:, i]

            if masks is not None:
                score = torch.where(masks[:, i: i+1], _score, score)
                index = torch.where(masks[:, i: i+1], _index, dummy)
            else:
                score, index = _score, _index
            traj_tokens.append(index)

        # now running the back-tracing and find the best
        best_score, best_index = score.max(dim=1)
        finalized_tokens.append(best_index[:, None])
        finalized_scores.append(best_score[:, None])

        for idx, scs in zip(reversed(traj_tokens), reversed(traj_scores)):
            previous_index = finalized_tokens[-1]
            finalized_tokens.append(idx.gather(1, previous_index))
            finalized_scores.append(scs.gather(1, previous_index))

        finalized_tokens.reverse()
        # at this point, finalized tokens is a 3d tensor
        # finalized_tokens[0] is of batch_size x 1
        # containing the first token index for each batch
        # we do torch.cat to make it batch_size by seq_len
        # which is a sequence of token indices for each batch
        finalized_tokens = torch.cat(finalized_tokens, 1)

        # but remember, these are token indices, not the tokens themselves
        # beam_targets is a 3d tensor. The first dimension is the batch
        # for each batch, there is a 2d tensor of size seq_len x beam
        # it says which are the topk (k = beam) tokens for each position,
        # and we select those using our finalized tokens indices
        finalized_tokens = beam_targets.gather(2, finalized_tokens[:, :, None])[:, :, 0]

        finalized_scores.reverse()
        finalized_scores = torch.cat(finalized_scores, 1)
        finalized_scores[:, 1:] = finalized_scores[:, 1:] - finalized_scores[:, :-1]

        return finalized_scores, finalized_tokens