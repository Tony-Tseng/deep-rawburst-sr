import math
import torch
from torch.optim import Optimizer

class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

        return loss