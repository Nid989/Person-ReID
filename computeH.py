import torch
import numpy as np

def computeH(n, ib, ie):

    # compute the kernel matrix H
    # Input: 
        # n: the number of training examples 
        # ib: indices of the first terms in pairs.
        # ie: indices of the second terms in pairs.

    B = torch.zeros(n, 1)
    E = torch.zeros(n, 1)
    W = torch.zeros(n, n)
    
    

    return 