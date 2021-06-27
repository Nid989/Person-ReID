import numpy as np
import torch 
    
def projectPSD(M):
    #  project the matrix M to its cone of PSD
    #  INPUT
    #    M: a squared matrix
    #  OUTPUT
    #    M: the PSD matrix

    D, V = np.linalg.eig(M)
    V = np.real(V)
    d = np.real(D)
    d[d<=0] = np.finfo(np.float64).eps
    M = np.matmul(V, np.matmul(np.diag(d), V.T))
    return M