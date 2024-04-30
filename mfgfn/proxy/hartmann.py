import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.test_functions.synthetic import Hartmann as BotorchHartmann
from gflownet.proxy.base import Proxy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType


class Hartmann(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This is just a rough estimate of modes
        self.modes = [
            [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.9, 0.6, 0.1, 0.0],
            [0.3, 0.1, 0.4, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.4, 0.6, 0.0, 0.0],
            [0.4, 0.9, 0.6, 0.6, 0.3, 0.0],
        ]
        self.extrema = 3.32237


class MultiFidelityHartmann(Hartmann):
    def __init__(self, fid=None, cost=None, **kwargs):
        super().__init__(**kwargs)
        # maximization problem so negate = True
        self.task = AugmentedHartmann(negate=True)
        self.fid = fid
        self.cost = cost

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        fid_dict = {0: 0.5, 1: 0.75, 2: 1.0}
        fid = self.fid
        fid = fid_dict[fid]
        fidelity = (
            torch.ones((len(states), 1), device=self.device, dtype=self.float) * fid
        )
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        else:
            states = states.to(self.device).to(self.float)
        # if states.shape[1] != 6:
        # states = states[:, :6]
        state_fid = torch.cat([states, fidelity], dim=1)
        scores = self.task(state_fid)
        # scores = scores.unsqueeze(-1)
        return scores.to(self.device).to(self.float)


class SingleFidelityHartmann(Hartmann):
    def __init__(self, cost, **kwargs):
        super().__init__(**kwargs)
        # maximization problem so negate = True
        self.task = BotorchHartmann(negate=True)
        self.cost = cost

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        else:
            states = states.to(self.device).to(self.float)
        scores = self.task(states)
        return scores.to(self.device).to(self.float)
