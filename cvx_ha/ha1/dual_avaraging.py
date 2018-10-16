import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class DualAveraging(Optimizer):
    def __init__(self, params, lr=required, dual_projection=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, dual_projection=dual_projection)
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
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Dual variables
                    if group['dual_projection'] is not None:
                        state['theta'] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['dual_projection'] is not None:
                    # 1) theta_{t+1} = theta_t + grad_t
                    state['theta'].add_(d_p)
                    # 2) x_{t+1} = dual_proj(theta_{t+1}, lr_t)
                    p.data.set_(group['dual_projection'](state['theta'], group['lr']))
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss
