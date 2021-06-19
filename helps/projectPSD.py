import torch 
    
def projectPSD(M):
    #  project the matrix M to its cone of PSD
    #  INPUT
    #    M: a squared matrix
    #  OUTPUT
    #    M: the PSD matrix

    D, V = torch.eig(M) # computes the eigenvectors and eigenvalues corresponding to M
    V = torch.real(V) # removes the complex part from matrix 'V' containing column eigen-vectors
    
    return