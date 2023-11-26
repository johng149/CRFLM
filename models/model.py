import torch
from torch.nn import Module, Linear, LayerNorm, ModuleList
from .helix import Helix, Endcap, Downsizer, SingleStrand
from .reader import Reader
from .phm import phm
from .crf import CRF

class CustomModel(Module):

    def __init__(self, embedding_dim, num_heads, vocab_size, num_helix_layers=2, num_single_strand_layers=2, phm_factor=4, lm_head_phm_factor=2):
        super().__init__()
        self.kwargs = {
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'vocab_size': vocab_size,
            'num_helix_layers': num_helix_layers,
            'num_single_strand_layers': num_single_strand_layers,
            'phm_factor': phm_factor,
            'lm_head_phm_factor': lm_head_phm_factor
        }
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.num_helix_layers = num_helix_layers
        self.num_single_strand_layers = num_single_strand_layers
        self.phm_factor = phm_factor

        self.reader = Reader(vocab_size, embedding_dim)
        self.helixes = ModuleList([Helix(embedding_dim, num_heads, phm_factor) for _ in range(num_helix_layers)])
        self.endcap = Endcap(embedding_dim, num_heads, phm_factor)
        self.single_strands = ModuleList([SingleStrand(embedding_dim, num_heads, phm_factor) for _ in range(num_single_strand_layers)])
        self.norm = LayerNorm(embedding_dim)

    def forward(self, query, context):
        # input is seq_len x batch_size
        # output is seq_len x batch_size x vocab_size
        query = self.reader(query)
        context = self.reader(context)
        for helix in self.helixes:
            query, context = helix(query, context)
        context = self.endcap(query, context)
        for single_strand in self.single_strands:
            context = single_strand(context)
        result = self.norm(context)
        return result.transpose(0,1)
    
class CRFModel(Module):

    def __init__(self, model, vocab_size, beam, low_rank, padding_idx):
        # we need the padding idx to make it so generating padding idx
        # does not contribute to the score.
        # remember, we are calculating the numerator and denominator
        # of a sequence for the CRF and we are trying to maximize
        # the probability of the sequence that we are given and produced
        # the numerator
        # by masking padding idx, we are making it so padding tokens do
        # not contribute to the score
        # to skip this step, set the padding idx to a value that is
        # not in the vocab
        super().__init__()
        self.kwargs = {
            "vocab_size": vocab_size,
            "beam": beam,
            "low_rank": low_rank,
            "padding_idx": padding_idx
        }
        self.vocab_size = vocab_size
        self.model = model
        self.crf = CRF(vocab_size, beam, low_rank)
        self.padding_idx = padding_idx
        self.lm_head = Linear(model.embedding_dim, vocab_size)

    def forward(self, query, context, target):
        modelout = self.model(query, context)
        batch_size, seq_len, embed_dim = modelout.shape
        modelout = modelout.transpose(0,1)
        logits = self.lm_head(modelout.contiguous().view(-1, embed_dim)).view(seq_len, batch_size, self.vocab_size)
        logits = logits.transpose(0,1)
        mask = ~target.eq(self.padding_idx)
        crf_losses = self.crf(logits, target, mask) * -1
        return logits, crf_losses
    
    def inference(self, query, context):
        modelout = self.model(query, context)
        batch_size, seq_len, embed_dim = modelout.shape
        modelout = modelout.transpose(0,1)
        logits = self.lm_head(modelout.contiguous().view(-1, embed_dim)).view(seq_len, batch_size, self.vocab_size)
        logits = logits.transpose(0,1)
        return self.crf.viterbi_decode(logits)