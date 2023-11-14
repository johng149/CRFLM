from torch.nn import Module, Linear, LayerNorm, ModuleList
from .helix import Helix, Endcap, Downsizer, SingleStrand
from .reader import Reader
from .phm import phm
from .crf import CRF

class CustomModel(Module):

    def __init__(self, embedding_dim, num_heads, target_len, vocab_size, num_helix_layers=2, num_single_strand_layers=2, phm_factor=4, lm_head_phm_factor=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.target_len = target_len
        self.vocab_size = vocab_size
        self.num_helix_layers = num_helix_layers
        self.num_single_strand_layers = num_single_strand_layers
        self.phm_factor = phm_factor

        self.reader = Reader(vocab_size, embedding_dim)
        self.helixes = ModuleList([Helix(embedding_dim, num_heads, phm_factor) for _ in range(num_helix_layers)])
        self.endcap = Endcap(embedding_dim, num_heads, phm_factor)
        self.downsizer = Downsizer(embedding_dim, num_heads, target_len, phm_factor)
        self.single_strands = ModuleList([SingleStrand(embedding_dim, num_heads, phm_factor) for _ in range(num_single_strand_layers)])
        self.norm = LayerNorm(embedding_dim)
        self.out = phm(lm_head_phm_factor, embedding_dim, vocab_size)

    def forward(self, query, context):
        # input is seq_len x batch_size
        # output is seq_len x batch_size x vocab_size
        query = self.reader(query)
        context = self.reader(context)
        for helix in self.helixes:
            query, context = helix(query, context)
        context = self.endcap(query, context)
        x = self.downsizer(context)
        for single_strand in self.single_strands:
            x = single_strand(x)
        result = self.norm(x)
        result = self.out(result).transpose(0,1)
        return result
    
class CRFModel(Module):

    def __init__(self, model, vocab_size, beam, low_rank, padding_idx):
        super().__init__()
        self.model = model
        self.crf = CRF(vocab_size, beam, low_rank)
        self.padding_idx = padding_idx

    def forward(self, query, context, target):
        logits = self.model(query, context)
        mask = ~target.eq(self.padding_idx)
        crf_losses = self.crf(logits, target, mask) * -1
        return logits, crf_losses
    
    def inference(self, query, context):
        logits = self.model(query, context)
        return self.crf.viterbi_decode(logits)