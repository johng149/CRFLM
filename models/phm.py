# based on https://arxiv.org/abs/2102.08597
# with code from https://github.com/MehmetBarutcu/PHM-Paper-Implementation
# and https://github.com/rabeehk/compacter/blob/main/seq2seq/hypercomplex/kronecker.py
# a Parameterized Hypercomplex Multiplication layer

import torch
from torch.nn import Parameter, Module
from torch.nn.init import xavier_uniform_

class KroneckerLinear(Module):

    def __init__(self, factor, in_dim, out_dim):
        # only works if both in_dim and out_dim are divisible by factor
        assert in_dim % factor == 0 and out_dim % factor == 0
        super().__init__()
        self.factor = factor
        self.in_dim = in_dim
        self.out_dim = out_dim

        # from https://github.com/MehmetBarutcu/PHM-Paper-Implementation

        self.a = torch.zeros((self.factor, self.factor, self.factor))
        self.a = Parameter(xavier_uniform_(self.a))

        self.s = torch.zeros((self.factor, self.out_dim//self.factor, self.in_dim//self.factor)) 
        self.s = Parameter(xavier_uniform_(self.s))
    
    def kronecker_product_einsum_batched(self, A, B):
        # from https://github.com/rabeehk/compacter/blob/main/seq2seq/hypercomplex/kronecker.py
        """
        Batched Version of Kronecker Products
        :param A: has shape (b, a, c)
        :param B: has shape (b, k, p)
        :return: (b, ak, cp)
        """
        res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                        A.size(1)*B.size(1),
                                                        A.size(2)*B.size(2))
        return res

def is_worth_it(factor, in_dim, out_dim):
    """
    Parameterized Hypercomplex Multiplication can potentially
    decrease the number of parameters by `1/factor`, however
    there is no such thing as free lunch. By doing this, we
    get a factor cubed term, and so there is a break even
    point where it is not worth it
    """

    phm_params = factor**3 + (in_dim * out_dim) / factor
    linear_params = in_dim * out_dim
    return phm_params < linear_params

def doable(factor, in_dim, out_dim):
    """
    Cannot create a PHM with arbitrary factor, in_dim, and out_dim,
    it must be that both in_dim and out_dim are divisible by factor
    """
    return in_dim % factor == 0 and out_dim % factor == 0

def phm(factor, in_dim, out_dim):
    """
    Create a PHM layer with the given factor, in_dim, and out_dim
    if it is doable and worth it, otherwise create a linear layer
    """
    if doable(factor, in_dim, out_dim) and is_worth_it(factor, in_dim, out_dim):
        return KroneckerLinear(factor, in_dim, out_dim)
    else:
        return torch.nn.Linear(in_dim, out_dim)