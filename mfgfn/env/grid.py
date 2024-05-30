import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gflownet.envs.grid import Grid as GflowNetGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType

from mfgfn.env.base import GFlowNetEnv
from mfgfn.proxy.branin import MultiFidelityBranin, SingleFidelityBranin


class Grid(GFlowNetEnv, GflowNetGrid):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        elif "state" in self.proxy_state_format:
            # As We want it to be compatible with both state_fidId and state formats
            # state is for when proxy is GP so fidelities are fractional
            # state_fidIdx is for the multifidelity GP where an index kernel is used
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
            # Assumes that the oracle is always Branin
            self.statebatch2oracle = self.statebatch2state
            self.statetorch2oracle = self.statetorch2state
        else:
            raise NotImplementedError(
                f"Proxy state format {self.proxy_state_format} not implemented"
            )

    def statetorch2readable(self, state, alphabet={}):
        """
        Dataset Handler in activelearning deals only in tensors. This function converts the tesnor to readble format to save the train dataset
        """
        assert torch.eq(state.to(torch.long), state).all()
        state = state.to(torch.long)
        state = state.detach().cpu().numpy()
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def statebatch2state(self, state_batch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        if isinstance(state_batch, torch.Tensor) == False:
            state_batch = torch.tensor(state_batch)
        return self.statetorch2state(state_batch)

    def statetorch2state(self, state_torch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        if (
            isinstance(self.oracle, MultiFidelityBranin) == True
            or isinstance(self.oracle, SingleFidelityBranin) == True
        ):
            grid_size = self.length
            state_torch = state_torch / (grid_size - 1)
            state_torch[:, 0] = state_torch[:, 0] * self.rescale - 5
            state_torch[:, 1] = state_torch[:, 1] * self.rescale
        else:
            state_torch = state_torch / self.rescale
        return state_torch.to(self.float).to(self.device)

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, oracle=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if title == None:
            title = "Scores of Sampled States"
        if oracle is None:
            oracle = self.oracle
        if scores is None:
            if states is None or len(states) == 0:
                return None
            if isinstance(states, torch.Tensor) == False:
                states = torch.tensor(states, device=self.device, dtype=self.float)
            oracle_states = self.statetorch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def plot_samples_frequency(self, samples, ax=None, title=None, rescale=1):
        """
        Plot 2D histogram of samples.
        """
        if self.n_dim > 2:
            return None
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if len(samples) == 0:
            return None
        # assuming the first time this function would be called when the dataset is created
        if self.rescale == None:
            self.rescale = rescale
        # make a list of integers from 0 to n_dim
        if self.rescale != 1:
            step = int(self.length / self.rescale)
        else:
            step = 1
        ax.set_xticks(np.arange(start=0, stop=self.length, step=step))
        ax.set_yticks(np.arange(start=0, stop=self.length, step=step))
        # check if samples is on GPU
        if torch.is_tensor(samples) and samples.is_cuda:
            samples = samples.detach().cpu()
        states = np.array(samples).astype(int)
        grid = np.zeros((self.length, self.length))
        if title == None:
            ax.set_title("Frequency of Coordinates Sampled")
        else:
            ax.set_title(title)
        # TODO: optimize
        for state in states:
            grid[state[0], state[1]] += 1
        im = ax.imshow(grid)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def initialize_dataset(self, config, n_samples, resume, **kwargs):
        train_scores = []
        test_scores = []
        train_samples = []
        test_samples = []

        if config.oracle_dataset is not None:
            if config.oracle_dataset.train is not None:
                train = pd.read_csv(config.oracle_dataset.train.path)
                train_samples = train["samples"].values.tolist()
                if config.oracle_dataset.train.get_scores == False:
                    train_scores = train["scores"].values.tolist()

            if config.oracle_dataset.test is not None:
                test = pd.read_csv(config.oracle_dataset.test.path)
                test_samples = test["samples"].values.tolist()
                if config.oracle_dataset.test.get_scores == False:
                    test_scores = test["energies"].values.tolist()

        if train_samples == [] and test_samples == []:
            states = torch.tensor(
                self.get_uniform_terminating_states(n_samples), dtype=self.float
            )
        else:
            # samples = train_samples + test_samples
            train_states = [
                torch.tensor(self.readable2state(sample)) for sample in train_samples
            ]
            test_states = [
                torch.tensor(self.readable2state(sample)) for sample in test_samples
            ]
            states = torch.stack(train_states + test_states)
            # states = torch.cat([train_states, test_states])
        scores = train_scores + test_scores
        if scores == []:
            states_oracle_input = states.clone()
            state_oracle = self.statetorch2oracle(states_oracle_input)
            scores = self.oracle(state_oracle)

        if resume == False:
            return states, scores
        else:
            return train_states, train_scores, test_states, test_scores
        # return states, scores
