import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from scipy.optimize import minimize
from .projections import exp_quick_projection


class BregmanProjectionUpdate(object):
    def __init__(self, h=None, dh=None, opt_kwargs=None, x0=None, torch_compatible=False, closed_form=None):
        self.closed_form = closed_form
        self._h = h
        self._dh = dh
        self.opt_kwargs = opt_kwargs or dict()
        self.opt_kwargs.setdefault('constraints', None)
        self.opt_kwargs.setdefault('bounds', None)
        self.x0 = x0
        self.torch_compatible = torch_compatible
        if (
                (self.opt_kwargs['constraints'] or self.opt_kwargs['bounds'])
                and self.x0 is None
                and closed_form is None
        ):
            raise ValueError('Need x0 for constraints without closed form solution')
        elif (
                closed_form is None
                and (self._dh is None or self._h is None)
        ):
            raise ValueError('Need either closed form updates or distance function with its derivative')

    def h(self, x):
        if self.torch_compatible:
            return self._h(x)
        else:
            x = np.asarray(x)
            return torch.tensor(self._h(x))

    def dh(self, x):
        if self.torch_compatible:
            return self._dh(x)
        else:
            x = np.asarray(x)
            return torch.tensor(self._dh(x))

    def _fenchel_conjugate(self, theta):
        def problem(x):
            return -(x.dot(theta) - self._h(x))

        def grad(x):
            return -theta + self._dh(x)
        if self.x0 is None:
            self.x0 = theta.copy()
        solution = minimize(problem, x0=self.x0, jac=grad, **self.opt_kwargs)
        self.x0[:] = solution['x'].copy()
        return solution

    def dh_star(self, theta):
        x = self._fenchel_conjugate(theta.cpu().data.numpy())['x']
        return torch.tensor(x).type(theta.type()).to(theta.device)

    def h_star(self, theta):
        f = -self._fenchel_conjugate(theta.cpu().data.numpy())['fun']
        return torch.tensor(f).type(theta.type()).to(theta.device)

    def update_inplace(self, x, theta, grad, lr):
        if self.closed_form is None:
            theta.data.add_(-lr, grad)
            x.data.set_(self.dh_star(theta.data))
            theta.data.set_(self.dh(x.data))
        else:
            x.data.set_(self.closed_form(x, lr*grad))


brehman_exp_simplex_projection = BregmanProjectionUpdate(closed_form=exp_quick_projection)


class MirrorDescent(Optimizer):
    def __init__(self, params, brehman_projection, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, brehman_projection=brehman_projection)
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
                state = self.state[p]
                if len(state) == 0:
                    if group['brehman_projection'].closed_form is None:
                        # Dual variables
                        state['theta'] = group['brehman_projection'].dh(p.data)
                    else:
                        state['theta'] = None
                d_p = p.grad.data
                group['brehman_projection'].update_inplace(p, state['theta'], d_p, group['lr'])
        return loss
