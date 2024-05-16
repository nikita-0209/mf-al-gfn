from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from gflownet.utils.logger import Logger


class AL_Logger(Logger):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config,
        ckpts: dict,
        logdir: dict,
        **kwargs,
    ):
        super().__init__(config, logdir=logdir, checkpoints=ckpts.policy, **kwargs)
        self.proxy_period = (
            np.inf
            if ckpts.regressor.period == None or ckpts.regressor.period == -1
            else ckpts.regressor.period
        )
        self.data_dir = self.logdir / logdir.data
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def set_proxy_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.proxy_ckpt_path = None
        else:
            self.proxy_ckpt_path = self.ckpts_dir / f"{ckpt_id}"

    def save_proxy(self, model, optimizer, final, epoch):
        if not epoch % self.proxy_period or final:
            if final:
                ckpt_id = "final"
            else:
                ckpt_id = "epoch{:03d}".format(epoch)
            if self.proxy_ckpt_path is not None:
                stem = Path(
                    self.proxy_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                )
                path = self.proxy_ckpt_path.parent / stem
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    path,
                )

    def log_dataset_stats(self, train_stats, test_stats):
        if not self.do.online:
            return
        for key in train_stats.keys():
            self.log_metric("train_" + key, train_stats[key], use_context=False)
            if test_stats is not None:
                self.log_metric("test_" + key, test_stats[key], use_context=False)

    def set_data_path(self, data_path: str = None):
        if data_path is None:
            self.data_path = None
        else:
            self.data_path = self.data_dir / f"{data_path}"

    def save_dataset(self, dataset, type):
        if self.data_path is not None:
            data = pd.DataFrame(dataset)
            # sort dataframe by energies column in descending order
            data = data.sort_values(by="energies", ascending=False)
            if type == "sampled":
                type = type + "_iter" + self.context
            name = Path(self.data_path.stem + "_" + type + ".csv")
            path = self.data_path.parent / name
            data.to_csv(path)
        if self.do.online:
            # return
            wandb_path = Path(
                self.wandb.run.dir + "/" + self.data_path.stem + "_" + type + ".csv"
            )
            data.to_csv(wandb_path)

    def log_figure(self, key, fig, use_context):
        if not self.do.online and fig is not None:
            plt.close()
            return
        if use_context:
            key = self.context + "/" + key
        if fig is not None:
            try:
                figimg = self.wandb.Image(fig)
                self.wandb.log({key: figimg})
            except:
                pass
            plt.close()

    def define_metric(self, metric, step_metric=None):
        if not self.do.online:
            return
        wandb.define_metric(metric, step_metric=step_metric)
