import torch
import numpy as np

def computeH(n, ib, ie):

    # compute the kernel matrix H
    # Input: 
        # n: the number of training examples 
        # ib: indices of the first terms in pairs.
        # ie: indices of the second terms in pairs.

    ib = ib.type(torch.int64)
    ie = ie.type(torch.int64)

    B = torch.zeros(n, 1)
    E = torch.zeros(n, 1)
    W = torch.zeros(n, n)
    # Tensor of ones with shape [ib.shape[0], 1] ~ [316, 1]
    o = torch.ones(ib.shape[0], 1)
    B[ib] = o
    E[ie] = o

    B = torch.diag(B.view(n))
    E = torch.diag(E.view(n))

    Calc_W(ib, ie, W)

    # H = B - W.T - W + E
    H = B + E - W.T - W

    return H

def Calc_W(ib, ie, W):
    for i in range(0, ib.shape[0]):
        W[ib[i], ie[i]] = W[ib[i], ie[i]] + 1