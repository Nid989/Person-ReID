import torch
import numpy as np
from kernelmatrix import kernelmatrix
from helps.kernel import kernel
from helps.calcMCMC import calcMCMC 

def load_matfile(file_name):
    return np.load(file_name, allow_pickle=True)

def get_tensor(var, key, dtype=True):
    return torch.from_numpy((var.item().get(key)-1).astype('float64')) if dtype else torch.from_numpy(var.item().get(key))

# TODO rewrite later in more sensible and logical way.
def ret_logical(shape):
    temp_a = torch.ones(shape)
    temp_b = torch.cat((torch.ones(shape//2), torch.zeros(shape//2)))
    return torch.logical_and(temp_a, temp_b)
    
def change_view(tensor):
    return tensor.view(1, tensor.shape[0])  

if __name__ == "__main__":
    
    mat = load_matfile("viper//viper_features.npy")

    EPS   = 0.001; # the regularization constant
    sigma = 2**-16; # the kernel width
    N     = 632;  # number of persons
    d     = 100;  # number of features

    idxa = get_tensor(mat, 'idxa') # index of images in the first set
    idxb = get_tensor(mat, 'idxb') # index of images in the second set
    X = get_tensor(mat, 'ux', dtype=False)[:d, :] # training set

    # draw random permutation
    perm = torch.randperm(N)

    # split in equal-sized train and test sets
    idxtrain = perm[:N//2]
    idxtest = perm[N//2:]

    # BEGIN k-kissme
    first_ind = change_view(torch.cat((idxa[0, idxtrain], idxa[0, idxtrain])))
    second_ind = change_view(torch.cat((idxb[0, idxtrain], idxb[0, idxtrain[torch.randperm(N//2)]])))
    matches = ret_logical(N)
    
    S = torch.cat((first_ind[:, matches], second_ind[:, matches])) # must-link constraints
    D = torch.cat((first_ind[:, torch.logical_not(matches)], second_ind[:, torch.logical_not(matches)])) # cannot-link constraints
    
    K  = kernelmatrix(X=X, X2=torch.tensor([]), sigma=sigma);       
    # print(K)

    M = kernel(EPS,S,D,K,1); # learn k-KISSME
    # print(torch.diagonal(M)[0:100])
    # END k-kissme

    idxa = idxa.type(torch.int64).T
    idxb = idxb.type(torch.int64).T

    # test rank-1 matching rate
    cmc = calcMCMC(torch.eye(d), X, idxa, idxb, idxtest)
    cmc_ = calcMCMC(M, K, idxa, idxb, idxtest)
    

    s = idxtest.shape[0]
    print('Rank-1 matching rate:\n')
    print('IDENTITY = {0}\nk-KISSME = {1}\n'.format(100*cmc[:, 1]/s, 100*cmc_[:, 1]/s))
