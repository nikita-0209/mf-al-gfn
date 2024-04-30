from abc import abstractmethod

import gpytorch
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.models.gp_regression_fidelity import (
    SingleTaskGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.settings import debug
from gpytorch.mlls import VariationalELBO
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

"""
Assumes that in single fidelity, fid =1
"""


class SingleTaskGPRegressor:
    def __init__(self, logger, device, dataset, maximize, **kwargs):

        self.logger = logger
        self.device = device

        # Dataset
        self.dataset = dataset
        self.n_fid = dataset.n_fid
        self.n_samples = dataset.n_samples

        # Logger
        self.progress = self.logger.progress
        self.target_factor = self.dataset.target_factor

    @abstractmethod
    def init_model(self, train_x, train_y):
        # m is output dimension
        # TODO: if standardize is the desired operation
        pass

    def fit(self):
        train = self.dataset.train_dataset
        train_x = train["states"]
        train_y = train["energies"].unsqueeze(-1)
        self.init_model(train_x, train_y)
        with debug(state=True):
            self.mll = fit_gpytorch_mll(self.mll)

    def get_predictions(self, env, states, denorm=False):
        """Input is states
        Proxy conversion happens within."""
        detach = True
        if isinstance(states, torch.Tensor) == False:
            states = torch.tensor(states, device=self.device, dtype=env.float)
            detach = False
        if states.ndim == 1:
            states = states.unsqueeze(0)
        states_proxy_input = states.clone()
        states_proxy = env.statetorch2proxy(states_proxy_input)
        model = self.model
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(states_proxy)
            y_mean = posterior.mean
            y_std = posterior.variance.sqrt()
        if detach == True:
            # we don't want to detach when called after AL round
            y_mean = y_mean.detach().cpu().numpy().squeeze(-1)
            y_std = y_std.detach().cpu().numpy().squeeze(-1)
        else:
            y_mean = y_mean.squeeze(-1)
            y_std = y_std.squeeze(-1)
        if denorm == True and self.dataset.normalize_data == True:
            y_mean = (
                y_mean
                * (
                    self.dataset.train_stats["max"].cpu()
                    - self.dataset.train_stats["min"].cpu()
                )
                + self.dataset.train_stats["min"].cpu()
            )
            y_mean = y_mean.squeeze(-1)
        return y_mean, y_std

    def get_metrics(self, y_mean, y_std, env, states):
        state_oracle_input = states.clone()
        if hasattr(env, "call_oracle_per_fidelity"):
            samples, fidelity = env.statebatch2oracle(state_oracle_input)
            targets = env.call_oracle_per_fidelity(samples, fidelity).detach().cpu()
        elif hasattr(env, "oracle"):
            samples = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle(samples).detach().cpu()
        targets = targets * self.target_factor
        targets_numpy = targets.detach().cpu().numpy()
        targets_numpy = targets_numpy
        rmse = np.sqrt(np.mean((y_mean - targets_numpy) ** 2))
        nll = (
            -torch.distributions.Normal(torch.tensor(y_mean), torch.tensor(y_std))
            .log_prob(targets)
            .mean()
        )
        return rmse, nll

    def plot_predictions(self, states, scores, length, rescale=1):
        n_fid = self.n_fid
        n_states = int(length * length)
        if states.shape[-1] == 3:
            states = states[:, :2]
            states = torch.unique(states, dim=0)
        # states = states[:n_states]
        width = (n_fid) * 5
        fig, axs = plt.subplots(1, n_fid, figsize=(width, 5))
        for fid in range(0, n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((length, length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * n_states : (fid + 1) * n_states
            ]
            if n_fid == 1:
                ax = axs
            else:
                ax = axs[fid]
            if rescale != 1:
                step = int(length / rescale)
            else:
                step = 1
            ax.set_xticks(np.arange(start=0, stop=length, step=step))
            ax.set_yticks(np.arange(start=0, stop=length, step=step))
            ax.imshow(grid_scores)
            if n_fid == 1:
                title = "GP Predictions"
            else:
                title = "GP Predictions with fid {}/{}".format(fid + 1, n_fid)
            ax.set_title(title)
            im = ax.imshow(grid_scores)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig

    def get_modes(self, states, env):
        num_pick = int((env.length * env.length) / 100) * 5
        percent = num_pick / (env.length * env.length) * 100
        print(
            "\nUser-Defined Warning: Top {}% of states with maximum reward are picked for GP mode metrics".format(
                percent
            )
        )
        state_oracle_input = states.clone()
        if hasattr(env, "call_oracle_per_fidelity"):
            samples, fidelity = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle[-1](samples).detach().cpu()
        elif hasattr(env, "oracle"):
            samples = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle(samples).detach().cpu()
        targets = targets * self.target_factor
        idx_pick = torch.argsort(targets, descending=True)[:num_pick].tolist()
        states_pick = states[idx_pick]
        if hasattr(env, "oracle"):
            self._mode = states_pick
        else:
            fidelities = torch.zeros((len(states_pick) * 3, 1)).to(states_pick.device)
            for i in range(self.n_fid):
                fidelities[i * len(states_pick) : (i + 1) * len(states_pick), 0] = i
            states_pick = states_pick.repeat(self.n_fid, 1)
            state_pick_fid = torch.cat([states_pick, fidelities], dim=1)
            self._mode = state_pick_fid

    def evaluate_model(self, env, do_figure=True):
        if env.n_dim > 2:
            return None, 0.0, 0.0, 0.0, 0.0
        states = torch.FloatTensor(env.get_all_terminating_states()).to("cuda")
        y_mean, y_std = self.get_predictions(env, states)
        rmse, nll = self.get_metrics(y_mean, y_std, env, states)
        if hasattr(self, "_mode") == False:
            self.get_modes(states, env)
        mode_mean, mode_std = self.get_predictions(env, self._mode)
        mode_rmse, mode_nll = self.get_metrics(mode_mean, mode_std, env, self._mode)
        if do_figure:
            figure1 = self.plot_predictions(states, y_mean, env.length, env.rescale)
            figure2 = self.plot_predictions(states, y_std, env.length, env.rescale)
            fig = [figure1, figure2]
        else:
            fig = None
        return fig, rmse, nll, mode_rmse, mode_nll


class MultiFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        fid_column = train_x.shape[-1] - 1
        self.model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            # fid column
            data_fidelity=fid_column,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        self.mll.to(train_x)


class SingleFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        self.model = SingleTaskGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        self.mll.to(train_x)
