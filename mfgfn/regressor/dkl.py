import copy
import math
import os
from typing import List

import gpytorch
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from gflownet.utils.common import set_device, set_float_precision
from gpytorch import lazify
from gpytorch.lazy import ConstantDiagLazyTensor
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.settings import cholesky_jitter
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import LongTensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiStepLR
from torchtyping import TensorType
from tqdm import tqdm

from mfgfn.model.mlm import mlm_eval_epoch as mlm_eval_epoch_lm
from mfgfn.model.mlm import sample_mask
from mfgfn.model.mlp import MLP
from mfgfn.model.regressive import RegressiveMLP
from mfgfn.model.shared_elements import check_early_stopping
from mfgfn.utils.dkl_utils import batched_call


class Tokenizer:
    def __init__(self, non_special_vocab):
        self.non_special_vocab = non_special_vocab
        if "[nop]" in non_special_vocab:
            padding_token = "[nop]"
        else:
            padding_token = "[PAD]"
        # skipped UNK and 0
        self.special_vocab = ["[EOS]", "[CLS]", "[MASK]"] + [padding_token]
        special_dict = dict.fromkeys(self.special_vocab)
        non_special_dict = dict.fromkeys(self.non_special_vocab)
        special_vocab_dict = dict.fromkeys(
            x for x in special_dict if x not in non_special_dict
        )
        self.special_vocab = list(special_vocab_dict.keys())
        # remove duplicates (like if [EOS] and [PAD] were already in the dataset)
        self.full_vocab = self.non_special_vocab + self.special_vocab
        self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}
        # padding_token = "[nop]"
        masking_token = "[MASK]"
        bos_token = "[CLS]"
        eos_token = "[EOS]"
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]
        self.sampling_vocab = non_special_vocab
        self.special_idxs = [
            self.padding_idx,
            self.bos_idx,
            self.eos_idx,
            self.masking_idx,
        ]

    def transform(self, sequence_tensor):
        # for each tensor in tensor of tensors
        # find the index of the first [PAD] token
        # replace that index with [EOS] token
        index = [
            (
                torch.where(sequence == self.padding_idx)[0][0]
                if sequence[-1] == self.padding_idx
                else len(sequence)
            )
            for sequence in sequence_tensor
        ]
        # Padding element is added to all sequences so that [EOS] token can be added post the sequence, i.e,
        # at index 50 (assuming 0 indexing) if the sequence is of max length 50
        pad_tensor = (
            torch.ones(sequence_tensor.shape[0], 1).long().to(sequence_tensor)
            * self.padding_idx
        )
        sequence_tensor = torch.cat([sequence_tensor, pad_tensor], dim=-1)
        eos_tensor = (
            torch.ones(sequence_tensor.shape[0], 1).long().to(sequence_tensor)
            * self.eos_idx
        )
        index = LongTensor(index).to(sequence_tensor.device).unsqueeze(1)
        sequence_tensor = sequence_tensor.scatter(1, index, eos_tensor)
        bos_tensor = (
            torch.ones(sequence_tensor.shape[0], 1).long().to(sequence_tensor)
            * self.bos_idx
        )
        sequence_tensor = torch.cat([bos_tensor, sequence_tensor], dim=-1)
        return sequence_tensor.long()


class DeepKernelRegressor:
    def __init__(
        self,
        logger,
        device,
        dataset,
        config_model,
        config_env,
        surrogate,
        float_precision,
        tokenizer,
        checkpoint,
        encoder_obj="mlm",
        optim="adam",
        lr_sched_type="plateau",
        batch_size=32,
        **kwargs,
    ):
        self.logger = logger
        self.progress = self.logger.progress

        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

        # Dataset
        self.dataset = dataset
        self.n_fid = self.dataset.n_fid

        self.tokenizer = tokenizer
        self.language_model = hydra.utils.instantiate(
            config_model,
            tokenizer=self.tokenizer,
            device=self.device,
            float_precision=self.float,
            config_env=config_env,
            n_fid=self.n_fid,
            _recursive_=False,
        )
        is_fid_param = self.language_model.is_fid_param
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate = hydra.utils.instantiate(
            surrogate,
            tokenizer=tokenizer,
            encoder=self.language_model,
            device=self.device,
            float_precision=self.float,
            n_fid=self.n_fid,
            is_fid_param_nn=is_fid_param,
        )

        self.batch_size = self.surrogate.bs
        if checkpoint:
            self.logger.set_proxy_path(checkpoint)

        if isinstance(self.language_model, MLP) or isinstance(
            self.language_model, RegressiveMLP
        ):
            self.mlm_eval_epoch = self.language_model.eval_epoch
        else:
            self.mlm_eval_epoch = mlm_eval_epoch_lm

        self.optim = optim
        self.lr_sched_type = lr_sched_type

    def initialize_surrogate(self, X_train, Y_train):
        if (
            hasattr(self.surrogate, "init_inducing_points")
            and self.surrogate.num_inducing_points <= X_train.shape[0]
        ):
            print("\n---- initializing GP variational params ----")
            self.surrogate.eval()
            self.surrogate.requires_grad_(False)
            init_features = torch.cat(
                batched_call(self.surrogate.get_features, X_train, self.batch_size)
            )
            self.surrogate.train()
            self.surrogate.init_inducing_points(init_features)
            self.surrogate.initialize_var_dist_sgpr(
                init_features,
                Y_train.to(init_features),
                noise_lb=1.0,
            )
            if self.progress:
                print("variational initialization successful")

        self.mll = VariationalELBO(
            self.surrogate.likelihood, self.surrogate.model, num_data=X_train.shape[0]
        )
        self.mll.to(self.surrogate.device)

    def gp_train_step(self, optimizer, inputs, targets, mll):
        self.surrogate.zero_grad(set_to_none=True)
        self.surrogate.clear_cache()  # possibly unnecessary
        features = self.surrogate.get_features(
            inputs.to(self.surrogate.device), self.batch_size, transform=False
        )

        dtype = features.dtype if (features.dtype is not torch.bool) else torch.float
        targets = self.surrogate.reshape_targets(targets).to(
            features.device, dtype=dtype
        )

        output = self.surrogate.forward(features)
        loss = -mll(output, targets).mean()

        loss.backward()
        optimizer.step()
        return loss

    def fit(self):
        select_crit_key = "test_nll"
        X_train = self.dataset.train_dataset["states"]
        Y_train = self.dataset.train_dataset["energies"]
        Y_train = self.surrogate.reshape_targets(Y_train)
        Y_train = Y_train.to(dtype=list(self.surrogate.parameters())[0].dtype)

        train_loader, test_loader = self.dataset.get_dataloader()

        print("\n---- preparing checkpoint ----")
        self.surrogate.eval()
        self.surrogate.requires_grad_(False)
        self.surrogate.set_train_data(X_train, Y_train, strict=False)
        start_metrics = {}
        start_metrics.update(self.surrogate.evaluate(test_loader))
        start_metrics["epoch"] = 0
        if hasattr(self.language_model, "mask_ratio"):
            mask_ratio = self.language_model.mask_ratio
        else:
            mask_ratio = None
        start_metrics.update(
            self.mlm_eval_epoch(
                model=self.language_model,
                mask_ratio=mask_ratio,
                n_fid=self.n_fid,
                loader=test_loader,
            )
        )

        best_score = start_metrics.get(select_crit_key, None)
        best_score_epoch = 0
        self.surrogate.cpu()  # avoid storing two copies of the weights on GPU
        best_weights = copy.deepcopy(self.surrogate.state_dict())
        self.surrogate.to(self.surrogate.device)
        if best_score is not None and self.progress is True:
            print(f"\nstarting Test NLL: {best_score:.4f}")

        self.initialize_surrogate(X_train, Y_train)

        if hasattr(self.mll, "num_data"):
            self.mll.num_data = len(train_loader.dataset)

        best_loss, best_loss_epoch = None, 0
        stop = False

        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.surrogate.param_groups)
        else:
            optimizer = torch.optim.SGD(self.surrogate.param_groups)
        if self.lr_sched_type == "plateau":
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=math.ceil(self.surrogate.patience / 2.0),
                threshold=1e-3,
            )
        elif self.lr_sched_type == "step":
            lr_sched = MultiStepLR(
                optimizer,
                milestones=[
                    0.25 * self.surrogate.num_epochs,
                    0.5 * self.surrogate.num_epochs,
                    0.75 * self.surrogate.num_epochs,
                ],
                gamma=0.1,
            )

        print("\n---- fitting all SVGP params ----")
        pbar = tqdm(range(1, self.surrogate.num_epochs + 1), disable=not self.progress)
        for epoch_idx in pbar:

            metrics = {}
            avg_train_loss = 0.0
            avg_mlm_loss = 0.0
            avg_gp_loss = 0.0
            self.surrogate.train()
            if self.encoder_obj != "mlm":
                mlm_loss = torch.tensor(0.0, device=self.surrogate.device)

            for inputs, targets in train_loader:
                inputs = inputs.to(self.surrogate.device)
                targets = targets.to(self.surrogate.device)
                # train encoder through unsupervised MLM objective
                if self.encoder_obj == "mlm":
                    self.surrogate.encoder.requires_grad_(True)
                    mlm_loss = self.surrogate.encoder.train_step(
                        optimizer=optimizer,
                        input_batch=inputs,
                        target_batch=targets,
                        n_fid=self.n_fid,
                    )
                self.surrogate.requires_grad_(True)
                gp_loss = self.gp_train_step(optimizer, inputs, targets, self.mll)
                avg_train_loss += (mlm_loss.detach() + gp_loss.detach()) / len(
                    train_loader
                )
                avg_mlm_loss += mlm_loss.detach() / len(train_loader)
                avg_gp_loss += gp_loss.detach() / len(train_loader)

            lr_sched.step(avg_train_loss)

            metrics.update(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": avg_train_loss.item(),
                    "mlm_loss": avg_mlm_loss.item(),
                    "gp_loss": avg_gp_loss.item(),
                }
            )

            if (epoch_idx + 1) % self.surrogate.eval_period == 0:
                self.surrogate.requires_grad_(False)

                # update train features, use unaugmented train data for evaluation
                self.surrogate.eval()
                self.surrogate.set_train_data(X_train, Y_train, strict=False)
                metrics.update(self.surrogate.evaluate(test_loader))
                if self.encoder_obj == "mlm":
                    metrics.update(
                        self.mlm_eval_epoch(
                            model=self.language_model,
                            mask_ratio=mask_ratio,
                            n_fid=self.n_fid,
                            loader=test_loader,
                        )
                    )
            if self.progress:
                description = (
                    "Train Loss: {:.4f} | Test NLL: {:.4f} | Test RMSE {:.4f}".format(
                        avg_train_loss, metrics["test_nll"], metrics["test_rmse"]
                    )
                )
                pbar.set_description(description)

            # use validation NLL for model selection
            select_crit = metrics.get(select_crit_key, None)
            if self.surrogate.early_stopping and select_crit is not None:
                # and select_crit is not None:
                assert (
                    self.surrogate.holdout_ratio > 0.0
                ), "Need validation data for early stopping"
                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=self.surrogate,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=select_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=self.surrogate.patience,
                    save_weights=True,
                )
            metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

            # use train loss to determine convergence
            stop_crit_key = "test_nll"
            stop_crit = metrics.get(stop_crit_key, None)
            if stop_crit is not None:
                best_loss, best_loss_epoch, _, stop = check_early_stopping(
                    model=self.surrogate,
                    best_score=best_loss,
                    best_epoch=best_loss_epoch,
                    best_weights=None,
                    curr_score=stop_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=self.surrogate.patience,
                    save_weights=False,
                )
            metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

            log_prefix = "svgp"
            if len(log_prefix) > 0:
                metrics = {
                    "/".join((log_prefix, key)): val for key, val in metrics.items()
                }
            try:
                self.logger.log_metrics(metrics, use_context=True)
            except Exception:
                pass

            if stop:
                break

        if self.surrogate.early_stopping:
            print(f"\n---- loading checkpoint from epoch {best_score_epoch} ----")
            self.surrogate.load_state_dict(best_weights)
            print(f"---- best test NLL: {best_loss:.4f} ----")
            self.best_score = best_score
            self.best_loss = best_loss
        self.logger.save_proxy(
            self.surrogate, optimizer, epoch=best_score_epoch, final=True
        )
        self.surrogate.requires_grad_(False)
        self.surrogate.train()  # clear caches
        self.surrogate.clear_cache()
        self.surrogate.eval()
        self.surrogate.set_train_data(X_train, Y_train, strict=False)

    def get_predictions(self, env, states, denorm=False):
        self.surrogate.eval()
        self.surrogate.clear_cache()
        self.surrogate.requires_grad_(False)
        if isinstance(states[0], list):
            states = torch.tensor(states)
        elif isinstance(states, TensorType):
            pass
        elif isinstance(states[0], TensorType):
            states = torch.vstack(states)
        elif isinstance(states, List):
            states = torch.tensor(states)
            states = states.unsqueeze(0)
        states_proxy_input = states.clone()
        states_proxy = env.statetorch2proxy(states_proxy_input)
        y_mean, y_std, f_std = [], [], []
        with torch.no_grad():
            for batch in range(0, len(states_proxy), 32):
                states_proxy_batch = states_proxy[batch : batch + 32]
                features = self.surrogate.get_features(
                    states_proxy_batch.to(self.device), transform=None
                )
                f_dist = self.surrogate(features)
                y_dist = self.surrogate.likelihood(f_dist)
                f_std.append(f_dist.variance.sqrt().cpu())
                y_mean.append(y_dist.mean.cpu())
                y_std.append(y_dist.variance.sqrt().cpu())
        cat_dim = 0
        f_std = torch.cat(f_std, cat_dim).view(len(states), -1)
        y_mean = torch.cat(y_mean, cat_dim).view(len(states), -1)
        y_std = torch.cat(y_std, cat_dim).view(len(states), -1)
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

    def evaluate_model(self, env, do_figure):
        test_nll = self.best_score
        test_rmse = self.best_loss
        if hasattr(env, "n_dim") == False or env.n_dim > 2:
            return None, 0.0, 0.0, 0.0, 0.0
        states = torch.FloatTensor(env.get_all_terminating_states()).to("cuda")
        y_mean, y_std = self.get_predictions(env, states)
        if do_figure:
            figure1 = self.plot_predictions(states, y_mean, env.length, env.rescale)
            figure2 = self.plot_predictions(states, y_std, env.length, env.rescale)
            figure = [figure1, figure2]
        else:
            figure = None
        return figure, 0.0, 0.0, 0.0, 0.0

    def plot_predictions(self, states, scores, length, rescale=1):
        n_fid = self.n_fid
        n_states = int(length * length)
        if states.shape[-1] == 3:
            states = states[:, :2]
            states = torch.unique(states, dim=0)
        # states = states[:n_states]
        width = (n_fid) * 5
        fig, axs = plt.subplots(1, n_fid, figsize=(width, 5))
        scores = scores.squeeze(-1).detach().cpu().numpy()
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
