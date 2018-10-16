import torch


def simplex_projection(x, z=1.):
    n_features = x.size(0)
    u, _ = x.sort(descending=True)
    cssv = u.cumsum(0) - z
    ind = torch.arange(n_features).float() + 1.
    cond = (u - cssv / ind) > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = torch.relu(x - theta)
    return w


def simplex_dual_projection(theta, alpha):
    return simplex_projection(-theta*alpha)
