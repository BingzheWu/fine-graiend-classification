import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import torch.distributions as dists
class dp_sgd(optim.SGD):
    def __init__(self, params, lr = required, momentum = 0, weight_decay = 0.0):
        defaults = dict(lr = lr, momentum = momentum, weight_decay = weight_decay)
        super(dp_sgd, self).__init__(params, lr = lr, momentum = momentum, weight_decay = weight_decay)
        self.gauss = dists.laplace.Laplace(torch.tensor([0.0]), torch.tensor([0.01]))
        #self.normal = dists.normal.Normal(loc = torch.zeros(), scale)
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                gauss = dists.normal.Normal(loc = torch.zeros(d_p.size()), scale = 0.05*torch.ones(d_p.size()))
                noise = gauss.sample().cuda()
                d_p.add_(noise)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
