import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from botorch.test_functions.synthetic import Branin as BotorchBranin
from gflownet.proxy.base import Proxy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType


class Branin(Proxy):
    def __init__(self, **kwargs):
        """
        Modes compatible with 100x100 grid"""
        self.modes = [
            [12.4, 81.833],
            [54.266, 15.16],
            [94.98, 16.5],
        ]
        self.extrema = 0.397887
        super().__init__(**kwargs)

    def plot_true_rewards(self, env, ax, rescale):
        states = torch.FloatTensor(env.get_all_terminating_states()).to(self.device)
        states_oracle = states.clone()
        grid_size = env.length
        states_oracle = states_oracle / (grid_size - 1)
        states_oracle[:, 0] = states_oracle[:, 0] * rescale - 5
        states_oracle[:, 1] = states_oracle[:, 1] * rescale
        scores = self(states_oracle).detach().cpu().numpy()
        if hasattr(self, "fid"):
            title = "Oracle Energy (TrainY) with fid {}".format(self.fid)
        else:
            title = "Oracle Energy (TrainY)"
        # what the GP is trained on
        if self.maximize == False:
            scores = scores * (-1)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((env.length, env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores
        ax.set_xticks(
            np.arange(start=0, stop=env.length, step=int(env.length / rescale))
        )
        ax.set_yticks(
            np.arange(start=0, stop=env.length, step=int(env.length / rescale))
        )
        ax.imshow(grid_scores)
        ax.set_title(title)
        im = ax.imshow(grid_scores)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        return ax


class MultiFidelityBranin(Branin):
    def __init__(self, fid=None, cost=None, **kwargs):
        super().__init__(**kwargs)
        # minimisation problem so negate = False
        self.task = AugmentedBranin(negate=False)
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
        if states.shape[1] != 2:
            states = states[:, :2]
        state_fid = torch.cat([states, fidelity], dim=1)
        scores = self.task(state_fid)
        # scores = scores.unsqueeze(-1)
        return scores.to(self.device).to(self.float)


class SingleFidelityBranin(Branin):
    def __init__(self, cost, **kwargs):
        self.cost = cost
        super().__init__(**kwargs)
        # minimisation problem so negate = False
        self.task = BotorchBranin(negate=False)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        else:
            states = states.to(self.device).to(self.float)
        scores = self.task(states)
        return scores.to(self.device).to(self.float)
