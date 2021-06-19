import torch
import numpy as np
from kernelmatrix import kernelmatrix


def load_matfile(file_name):
    return np.load(file_name, allow_pickle=True)

def get_tensor(var, key, dtype=True):
    return torch.from_numpy(var.item().get(key).astype('float64')) if dtype else torch.from_numpy(var.item().get(key))

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
    print(K)
