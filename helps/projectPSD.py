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
    

    # D, V = torch.eig(M, eigenvectors=True) # computes the eigenvectors and eigenvalues corresponding to M
    # # remove complex part from both EigneValues and EigenVectors
    # # Pytorch implementation of torch.eig(M) computes matrix D: n x 2 
    # # where D[:,0] = real values and D[:,1] = complex values
    # # V: n x n only computes real valued eigenvectors
    # d = D[:, 0]
    # for idx, i in enumerate(d):
    #     if i<=0:
    #         d[idx] = torch.finfo(torch.float64).eps
    
    # D = torch.diag(d).type(torch.double)
    # V = V.type(torch.double)

    # # Eignevalue decomposition of any matrix 'A' is given by
    # # A = Vdiag(D)V'
    # M = torch.mm(V, torch.mm(D, V.T))
    # return M