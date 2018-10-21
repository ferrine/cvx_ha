import torch
import numpy as np


# http://www.mblondel.org/publications/mblondel-icpr2014.pdf
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


# https://bayen.eecs.berkeley.edu/sites/default/files/conferences/efficient_bregman_projections.pdf
# https://github.com/walidk/BregmanProjection/blob/master/bregman/projections.py

def exp_quick_projection(x, g, epsilon=1e-4):
    r"""Computes the Bregman projection, with exponential potential, of a vector x given a gradient vector g, using a
    randomized pivot method. The expected complexity of this method is O(d), where d is the size of x.
    Takes as input
    - the parameter epsilon of the exponential potential.
    - the current iterate x
    - the gradient vector (scaled by the step size) g

    Notes
    -----
    This exponential projection is essentially our simplex projection wrt h(x) = \sum x_i log(x_i)
    """
    d = len(x)
    y = (x + epsilon) * torch.exp(-g)
    J = range(0, d)
    S = 0
    C = 0
    while len(J) > 0:
        j = np.random.choice(J)
        pivot = y[j]
        JP = list(i for i in J if y[i] >= pivot)
        JM = list(i for i in J if y[i] < pivot)
        CP = len(JP)
        SP = sum(y[i] for i in JP)
        gamma = (1+epsilon*(C+CP)*pivot - epsilon*(S+SP))
        if gamma > 0:
            J = JM
            S += SP
            C += CP
        else:
            J = JP
    Z = S/(1+epsilon*C)
    return torch.relu(-epsilon+y/Z)
