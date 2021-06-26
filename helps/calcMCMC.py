import torch
from helps.sqdist import mahalanobis_dist, sqdist

def calcMCMC(M, data, idxa, idxb, idxtest):
    
    p = data[:, idxa[idxtest]]
    p = p.view(p.shape[0], -1)
    q = data[:, idxb[idxtest]]
    q = q.view(q.shape[0], -1)    
    dist = mahalanobis_dist(p, q, M)

    tmp = 0
    result = torch.zeros(1, dist.shape[1])
    for pairCounter in range(0, dist.shape[1]):
        distPair = dist[pairCounter, :]
        tmp, idx = torch.sort(distPair)
        result[:, idx==pairCounter] = result[:, idx==pairCounter] + 1

    # tmp = 0
    # for counter in range(1, result.shape[0]):
    #     result[:, counter] = result[:, counter] + tmp
    #     tmp = result[:, counter]
            
    return result