from torch.optim.optimizer import Optimizer, required


class ProjectedSGD(Optimizer):
    R"""
    A simple implementation of projected SGD gradient descent

    Parameters
    ----------
    params : iterable
        Parameters to optimize
    projection : callable
        Optional default projection
    lr : float
        Learning Rate

    Notes
    -----
    General setup assumes

    .. math::

        x_{t+1} &= \argmin_{x \in D}
            \left\{
                \|x-(x_{t-1} - \eta_{t}g_t)\|_2^2
            \right\}
                &= \Pi_D(x_{t-1} - \eta_{t}g_t)

    Where :math:`D` is a convex set and

    .. math::

        \Pi_D(x) = \argmin_{y \in D} \|x-y\|_2^2

    """

    def __init__(self, params, lr=required, projection=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, projection=projection)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters
        ----------
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                if group['projection'] is not None:
                    p.data.set_(group['projection'](p.data))
        return loss
