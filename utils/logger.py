from gflownet.utils.logger import Logger
import torch
from pathlib import Path
import numpy as np
import pandas as pd


class AL_Logger(Logger):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config,
        do,
        project_name,
        logdir,
        sampler,
        progress,
        lightweight,
        debug,
        proxy,
        run_name=None,
        tags=None,
    ):
        super().__init__(
            config,
            do,
            project_name,
            logdir,
            sampler,
            progress,
            lightweight,
            debug,
            run_name,
            tags,
        )
        self.proxy_period = (
            np.inf if proxy.period == None or proxy.period == -1 else proxy.period
        )
        self.data_dir = self.logdir / logdir.data
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def set_proxy_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.proxy_ckpt_path = None
        else:
            self.proxy_ckpt_path = self.ckpts_dir / f"{ckpt_id}"

    def save_proxy(self, model, final, epoch):
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
                torch.save(model.state_dict(), path)

    def log_dataset_stats(self, train_stats, test_stats):
        if not self.do.online:
            return
        for key, _ in train_stats.items():
            key = "Train_" + key
        for key, _ in test_stats.items():
            key = "Test_" + key

        self.log_metrics(train_stats, True)
        self.log_metrics(test_stats, True)

    def set_data_path(self, data_path: str = None):
        if data_path is None:
            self.data_path = None
        else:
            self.data_path = self.data_dir / f"{data_path}"

    def save_dataset(self, train_dataset, test_dataset):
        if self.data_path is not None:
            train = pd.DataFrame(train_dataset)
            test = pd.DataFrame(test_dataset)

            train_stem = Path(self.data_path.stem + "_train.csv")
            train_path = self.data_path.parent / train_stem

            test_stem = Path(self.data_path.stem + "_test.csv")
            test_path = self.data_path.parent / test_stem
            train.to_csv(train_path)
            test.to_csv(test_path)
