import torch

def sqdist(p, q):

    (d, pn) = p.shape
    (d, qn) = q.shape
    pmag = torch.sum(torch.square(p), 0, keepdim=True).double()
    qmag = torch.sum(torch.square(q), 0, keepdim=True).double()

    m = torch.mm(torch.ones(pn, 1).double(), qmag) + torch.mm(pmag.T, torch.ones(1, qn).double()) - 2 * torch.mm(p.T, q)
    return m

def mahalanobis_dist(p, q, A):

    (d, pn) = p.shape
    (d, qn) = q.shape
    Ap = torch.mm(A.double(), p)
    Aq = torch.mm(A.double(), q)
    pmag = torch.sum(p*Ap, 0, keepdim=True).double()
    qmag = torch.sum(q*Aq, 0, keepdim=True).double()

    m = torch.mm(torch.ones(pn, 1).double(), qmag) + torch.mm(pmag.T, torch.ones(1, qn).double()) - 2 * torch.mm(p.T, Aq)

    return m



    