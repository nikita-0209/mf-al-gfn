import copy
import math

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from gflownet.utils.common import set_device, set_float_precision
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from mfgfn.model.shared_elements import check_early_stopping
from mfgfn.regressor.dkl import DeepKernelRegressor


# TODO: Change this to a class that DOES NOT inherit from nn.Module
# Breaks line 348 in model/gp_models.py
class AsItIs(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_fid_param = False

    def forward(self, x):
        return x

    def get_features(self, x):
        return x


class SingleTaskSVGP(DeepKernelRegressor):
    def __init__(
        self,
        logger,
        device,
        dataset,
        surrogate,
        float_precision,
        checkpoint,
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

        self.language_model = AsItIs()

        self.surrogate_config = surrogate
        self.surrogate = hydra.utils.instantiate(
            surrogate,
            tokenizer=None,
            encoder=self.language_model,
            device=self.device,
            float_precision=self.float,
        )
        self.batch_size = self.surrogate.bs
        if checkpoint:
            self.logger.set_proxy_path(checkpoint)

        self.define_metric()

    def fit(self):
        """
        Different from DKL Regressor fit() as:
            1. Does not call mlm_eval_epoch()
        """

        select_crit_key = "test_nll"

        X_train = self.dataset.train_dataset["states"]
        Y_train = self.dataset.train_dataset["energies"]
        Y_train = self.surrogate.reshape_targets(Y_train)
        Y_train = Y_train.to(dtype=list(self.surrogate.parameters())[0].dtype)

        train_loader, test_loader = self.dataset.get_dataloader()

        print("\nPreparing checkpoint")
        self.surrogate.eval()
        self.surrogate.requires_grad_(False)
        self.surrogate.set_train_data(X_train, Y_train, strict=False)
        start_metrics = {}
        start_metrics.update(self.surrogate.evaluate(test_loader))
        start_metrics["epoch"] = 0

        best_score = start_metrics.get(select_crit_key, None)
        best_score_epoch = 0
        self.surrogate.cpu()  # avoid storing two copies of the weights on GPU
        best_weights = copy.deepcopy(self.surrogate.state_dict())
        self.surrogate.to(self.surrogate.device)
        if best_score is not None:
            print(f"starting Test NLL: {best_score:.4f}")

        self.initialize_surrogate(X_train, Y_train)

        if hasattr(self.mll, "num_data"):
            self.mll.num_data = len(train_loader.dataset)

        best_loss, best_loss_epoch = None, 0
        stop = False

        optimizer = torch.optim.Adam(self.surrogate.param_groups)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=math.ceil(self.surrogate.patience / 2.0),
            threshold=1e-3,
        )
        print("\nFitting all SVGP params ")
        pbar = tqdm(range(1, self.surrogate.num_epochs + 1))
        for epoch_idx in pbar:
            metrics = {}
            avg_train_loss = 0.0
            self.surrogate.train()
            for inputs, targets in train_loader:
                self.surrogate.requires_grad_(True)
                gp_loss = self.gp_train_step(optimizer, inputs, targets, self.mll)

                avg_train_loss += (gp_loss.detach()) / len(train_loader)

            lr_sched.step(avg_train_loss)

            metrics.update(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": avg_train_loss.item(),
                }
            )
            if epoch_idx % self.surrogate.eval_period == 0:
                self.surrogate.requires_grad_(False)

                self.surrogate.eval()
                self.surrogate.set_train_data(X_train, Y_train, strict=False)
                metrics.update(self.surrogate.evaluate(test_loader))
                if self.progress:
                    description = "Train Loss: {:.4f} | Test NLL: {:.4f} | Test RMSE {:.4f}".format(
                        avg_train_loss, metrics["test_nll"], metrics["test_rmse"]
                    )
                    pbar.set_description(description)

            select_crit = metrics.get(select_crit_key, None)
            if self.surrogate.early_stopping and select_crit is not None:
                # and select_crit is not None:
                # assert (
                #     self.surrogate.holdout_ratio > 0.0
                # ), "Need validation data for early stopping"
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

            # use test nll to determine convergence
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
        else:
            self.best_score = metrics["test_rmse"]
            self.best_loss = metrics["test_nll"]
        self.surrogate.requires_grad_(False)
        self.surrogate.train()  # clear caches
        self.surrogate.clear_cache()
        self.surrogate.eval()
        self.surrogate.set_train_data(X_train, Y_train, strict=False)


class SingleTaskMultiFidelitySVGP(SingleTaskSVGP):
    def __init__(
        self, logger, device, dataset, surrogate, float_precision, checkpoint, **kwargs
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

        self.language_model = AsItIs()
        is_fid_param_nn = self.language_model.is_fid_param
        self.surrogate_config = surrogate
        self.surrogate = hydra.utils.instantiate(
            surrogate,
            tokenizer=None,
            encoder=self.language_model,
            device=self.device,
            float_precision=self.float,
            n_fid=self.n_fid,
            is_fid_param_nn=is_fid_param_nn,
        )
        self.batch_size = self.surrogate.bs
        if checkpoint:
            self.logger.set_proxy_path(checkpoint)
