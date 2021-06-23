from numpy.core.fromnumeric import shape
from numpy.lib.format import header_data_from_array_1_0
from helps.projectPSD import projectPSD
from computeH import computeH
import torch
import numpy as np

def kernel(e, S, D, K, pmetric):
    #  compute the matrix C
    #  INPUT
    #    e: epsilon for the regularizer
    #    S: similar pairs in column
    #    D: dissimilarpars in column
    #    K: the kernel matrix
    #    pmetric: Is it a metric?
    #  OUTPUT
    # %   C: the matrix used for kernel

    n = K.shape[0]
    n1 = S.shape[1]
    n0 = D.shape[1]
    H0 = computeH(n, D[0, :], D[1, :])
    H1 = computeH(n, S[0, :], S[1, :])
    C = computeT(n0, e, H0, K)-computeT(n1, e, H1, K)

    if pmetric:
        C = projectPSD(C)

    return C

def computeT(n, e, H, K):
    # compute the inverse of (eI + (1-e))
    # TODO problem might be in t [specifically 2nd part] or rather in inverse maybe even last part
    t = torch.eye(K.shape[0]) + (1/(n*e))*torch.matmul(K.double(), H.double()) 
    Z = torch.inverse(t)
    scalar = (1/(n*e**2))
    I = scalar * torch.matmul(H.double(), Z)
    return I
