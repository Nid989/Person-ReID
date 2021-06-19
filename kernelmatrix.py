import torch 
import numpy as np

def kernelmatrix(X, X2, sigma):
    # rbf kernel (Radial basis function) 
    # Inputs:
    #       X:      (d x n) input data
    #       X2:     (d x m) test data
    #       sigma:  width of the RBF kernel
    #
    # Output:
    #       K: kernel matrix    
    
    X_sq = torch.square(X)
    n1sq = torch.sum(input=X_sq, dim=0, keepdim=True)
    n1 = n1sq.shape[1]

    if not torch.numel(X2):
        temp = torch.matmul(torch.ones(n1, 1).double(), n1sq.double())
        K = temp.T + temp - 2*(torch.matmul(X.T, X))
    else:
        X2_sq = torch.square(X2)
        n2sq = torch.sum(input=X2_sq, dim=0, keepdim=True)
        n2 = n2sq.shape[1]
        temp_a = torch.matmul(torch.ones(n2, 1).double(), n1sq.double())
        temp_b = torch.matmul(torch.ones(n1, 1).double(), n2sq.double())
        K = temp_a.T + temp_b - 2*(torch.matmul(X.T, X2))
        
    K = torch.exp(-torch.mul(K,sigma))
    return K
    