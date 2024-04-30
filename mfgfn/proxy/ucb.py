import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mfgfn.regressor.dkl import DeepKernelRegressor
from mfgfn.regressor.regressor import DropoutRegressor as SurrogateDropoutRegressor

from .botorch_models import ProxyBotorchUCB
from .dropout_regressor import DropoutRegressor


class UCB(DropoutRegressor):
    def __init__(self, kappa, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kappa = kappa

    def __call__(self, inputs):
        """
        Args
            inputs: batch x obs_dim
        Returns:
            score of dim (n_samples,), i.e, ndim=1"""
        self.regressor.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(self.device)
        outputs = self.regressor.forward_with_uncertainty(
            inputs, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = score.to(self.device).to(self.float)
        return score


class BotorchUCB(UCB):
    """
    1. NN based Proxy
    2. DKL based Proxy (Single Fidelity AMP)
    3. Stochastic VGP based on DKL Code
    """

    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler_config = sampler
        if isinstance(self.regressor, SurrogateDropoutRegressor):
            model = ProxyBotorchUCB(self.regressor, self.num_dropout_samples)
        else:
            model = self.regressor.surrogate
            model.eval()
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_config.num_samples]),
            seed=self.sampler_config.seed,
        )
        self.acqf = qUpperConfidenceBound(model=model, beta=self.kappa, sampler=sampler)
        self.out_dim = 1
        self.batch_size = 32

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.device, dtype=self.float)
        if isinstance(self.regressor, DeepKernelRegressor) == True:
            # TODO: Make uniform to get_features()
            if hasattr(self.regressor.language_model, "get_token_features"):
                (
                    input_tok_features,
                    input_mask,
                ) = self.regressor.language_model.get_token_features(inputs)
                _, pooled_features = self.regressor.language_model.pool_features(
                    input_tok_features, input_mask
                )
                inputs = pooled_features
            elif hasattr(self.regressor.language_model, "get_features"):
                inputs = self.regressor.language_model.get_features(inputs)
        inputs = inputs.unsqueeze(-2)
        acq_values = self.acqf(inputs)
        return acq_values


class GaussianProcessUCB(UCB):
    """
    Used for Single Fidelity Branin where the Proxy is a GP and Plotting of Acq Rewards can be done.
    """

    def __init__(self, sampler, env, logger, **kwargs):
        super().__init__(**kwargs)
        model = self.regressor.model
        model.eval()
        self.sampler_config = sampler
        self.logger = logger
        self.env = env
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_config.num_samples]),
            seed=self.sampler_config.seed,
        )
        self.acqf = qUpperConfidenceBound(model=model, beta=self.kappa, sampler=sampler)
        fig = self.plot_acquisition_rewards()
        if fig is not None:
            self.logger.log_figure("acquisition_rewards", fig, use_context=True)

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.device, dtype=self.float)
        else:
            inputs = inputs.to(self.device)
        inputs = inputs.unsqueeze(-2)
        acq_values = self.acqf(inputs)
        return acq_values

    def plot_acquisition_rewards(self, **kwargs):
        if self.env.n_dim != 2:
            return None
        if hasattr(self.env, "get_all_terminating_states") == False:
            return None
        states = torch.tensor(
            self.env.get_all_terminating_states(), dtype=self.float
        ).to(self.device)
        states_input_proxy = states.clone()
        states_proxy = self.env.statetorch2proxy(states_input_proxy)
        scores = self(states_proxy).detach().cpu().numpy()
        width = 5
        fig, axs = plt.subplots(1, 1, figsize=(width, 5))
        if self.env.rescale != 1:
            step = int(self.env.length / self.env.rescale)
        else:
            step = 1
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((self.env.length, self.env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores
        axs.set_xticks(
            np.arange(
                start=0,
                stop=self.env.length,
                step=step,
            )
        )
        axs.set_yticks(
            np.arange(
                start=0,
                stop=self.env.length,
                step=step,
            )
        )
        axs.imshow(grid_scores)
        axs.set_title("GP-UCB Reward")
        im = axs.imshow(grid_scores)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        plt.tight_layout()
        plt.close()
        return fig
