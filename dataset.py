from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from gflownet.utils.common import set_device, set_float_precision
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType
from typing import List
from utils.common import get_figure_plots


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data.detach().cpu()
        self.y_data = y_data.detach().cpu()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataHandler:
    """
    Intialises the train data using env-specific train function
    Scores data using oracle
    """

    def __init__(
        self,
        env,
        normalize_data,
        train_fraction,
        dataloader,
        path,
        logger,
        oracle,
        split,
        device,
        float_precision,
        is_mes,
        n_samples=None,
        # fidelity=None,
        rescale=None,
    ):
        self.env = env
        self.normalize_data = normalize_data
        self.train_fraction = train_fraction
        self.n_samples = n_samples
        self.dataloader = dataloader
        self.split = split
        self.path = path
        self.logger = logger
        # self.fidelity = fidelity
        self.progress = self.logger.progress
        self.target_factor = 1.0
        if oracle.maximize is False and is_mes is True:
            self.target_factor = -1.0
        self.logger.set_data_path(self.path.dataset)
        self.device = device
        if hasattr(env, "n_fid"):
            self.n_fid = env.n_fid
            self.sfenv = env.env
        else:
            self.n_fid = 1
            self.sfenv = env
        self.float = set_float_precision(float_precision)
        self.rescale = rescale
        self.initialise_dataset()

    def convert_to_tensor(self, data):
        """
        Converts a list of arrays to a tensor
        """
        if isinstance(data, TensorType):
            return data
        elif isinstance(data, List) and len(data) == 0:
            return torch.tensor(data, dtype=self.float, device=self.device)
        elif isinstance(data, List) and isinstance(data[0], TensorType):
            return torch.stack(data).to(self.device)
        elif isinstance(data, List):
            return torch.tensor(data, dtype=self.float, device=self.device)
        else:
            raise NotImplementedError(
                "Data type not recognized for conversion to tensor"
            )

    def scale_by_target_factor(self, data):
        if data is not None:
            data = data * self.target_factor
            indices = torch.where(data == -0.0)
            data[indices] = 0.0
        return data


    def initialise_dataset(self):
        # TODO: Modify to ensure validation set has equal number of points across fidelities
        """
        Loads the dataset as a dictionary
        OR
        Initialises the dataset using env-specific make_train_set function (returns a dataframe that is converted to a dictionary)

        - dataset['states']: list of arrays
        - dataset['energies']: list of float values

        If the dataset was initalised and save_data = True, the un-transformed (no proxy transformation) de-normalized data is saved as npy
        """
        if self.logger.resume:
            if self.n_fid > 1:
                self.path.type = "mf"
            else:
                self.path.type = "sf"
            self.path.oracle_dataset = {
                "train": {
                    "path": str(self.logger.data_path.parent / Path("data_train.csv")),
                    "get_scores": False,
                },
                "test": {
                    "path": str(self.logger.data_path.parent / Path("data_test.csv")),
                    "get_scores": False,
                },
            }
        if hasattr(self.env, "initialize_dataset"):
            if self.split == "given":
                logger_resume = True
            else:
                logger_resume = self.logger.resume
            state_score_tuple = self.env.initialize_dataset(
                self.path, self.n_samples, logger_resume
            )
            if self.logger.resume == True or self.split == "given":
                self.split = "given"
                train_states = state_score_tuple[0]
                train_scores = state_score_tuple[1]
                test_states = state_score_tuple[2]
                test_scores = state_score_tuple[3]
                states = torch.cat((train_states, test_states), dim=0)
                scores = torch.cat((train_scores, test_scores), dim=0)
            else:
                states = state_score_tuple[0]
                scores = state_score_tuple[1]
                test_scores = None
                train_scores = None
                test_states = None
                train_states = None
            if self.n_fid > 1:
                fidelities = states[:, -1].tolist()
                if train_states is not None:
                    train_fidelities = train_states[:, -1].tolist()
                    test_fidelities = test_states[:, -1].tolist()
                else:
                    train_fidelities = fidelities
                    test_fidelities = None
            else:
                fidelities = None
                train_fidelities = None
                test_fidelities = None

        else:
            raise NotImplementedError(
                "Dataset initialisation not implemented for this environment"
            )

        scores = self.scale_by_target_factor(scores)
        train_scores = self.scale_by_target_factor(train_scores)
        test_scores = self.scale_by_target_factor(test_scores)

        if self.split == "random":
            index = torch.randperm(len(states))
            train_index = index[: int(len(states) * self.train_fraction)]
            test_index = index[int(len(states) * self.train_fraction) :]
            train_states = states[train_index]
            test_states = states[test_index]
            if scores is not None:
                train_scores = scores[train_index]
                test_scores = scores[test_index]
            train_states = train_states.to(self.device)
            test_states = test_states.to(self.device)
            train_scores = train_scores.to(self.float).to(self.device)
            test_scores = test_scores.to(self.float).to(self.device)
        elif self.split == "all_train":
            train_states = states.to(self.device)
            train_scores = scores.to(self.device)
            test_states = torch.Tensor([])
            test_scores = torch.Tensor([])
        elif self.split == "given":
            assert train_states is not None
            assert test_states is not None
            assert train_scores is not None
            assert test_scores is not None
            train_states = train_states.to(self.device)
            test_states = test_states.to(self.device)
            train_scores = train_scores.to(self.float).to(self.device)
            test_scores = test_scores.to(self.float).to(self.device)
        else:
            raise ValueError("Split type not implemented")

        get_figure_plots(
            self.env,
            train_states,
            train_scores,
            train_fidelities,
            logger=self.logger,
            title="Initial Train Dataset",
            key="initial_train_dataset",
            use_context=True,
        )

        get_figure_plots(
            self.env,
            test_states,
            test_scores,
            test_fidelities,
            logger=self.logger,
            title="Initial Test Dataset",
            key="initial_test_dataset",
            use_context=True,
        )

        train_scores = self.scale_by_target_factor(train_scores)
        readable_train_samples = [
            self.env.statetorch2readable(sample) for sample in train_states
        ]
        readable_train_dataset = {
            "samples": readable_train_samples,
            "energies": train_scores.tolist(),
        }
        train_scores = self.scale_by_target_factor(train_scores)
        self.logger.save_dataset(readable_train_dataset, "train")
        self.train_dataset = {"states": train_states, "energies": train_scores}
        self.train_dataset, self.train_stats = self.preprocess(self.train_dataset)
        self.train_data = Data(
            self.train_dataset["states"], self.train_dataset["energies"]
        )

        if len(test_states) > 0:
            test_scores = self.scale_by_target_factor(test_scores)
            readable_test_samples = [
                self.env.statetorch2readable(sample) for sample in test_states
            ]
            readable_test_dataset = {
                "samples": readable_test_samples,
                "energies": test_scores.tolist(),
            }
            test_scores = self.scale_by_target_factor(test_scores)
            self.logger.save_dataset(readable_test_dataset, "test")
            self.test_dataset = {"states": test_states, "energies": test_scores}

            self.test_dataset, self.test_stats = self.preprocess(self.test_dataset)
            self.test_data = Data(
                self.test_dataset["states"], self.test_dataset["energies"]
            )
        else:
            self.test_dataset = None
            self.test_data = None
            self.test_stats = None

        # Log the dataset statistics
        self.logger.log_dataset_stats(self.train_stats, self.test_stats)
        if self.progress:
            print("\nProxy Dataset Statistics")
            print(
                "Train Data \n \t Mean Score:{:.2f} \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.train_stats["mean"],
                    self.train_stats["std"],
                    self.train_stats["min"],
                    self.train_stats["max"],
                )
            )
            if self.test_stats is not None:
                print(
                    "Test Data \n \t Mean Score:{:.2f}  \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                        self.test_stats["mean"],
                        self.test_stats["std"],
                        self.test_stats["min"],
                        self.test_stats["max"],
                    )
                )

    def preprocess(self, dataset):
        """
        - converts states to proxy space
        - normalizes the energies
        - shuffles the data
        - splits the data into train and test
        """
        states = dataset["states"]
        scores = dataset["energies"]
        state_input_proxy = states.clone()
        state_proxy = self.env.statetorch2proxy(state_input_proxy)
        # for when oracle is proxy and grid setup when oracle state is tensor
        if isinstance(state_proxy, tuple):
            state_proxy = torch.concat((state_proxy[0], state_proxy[1]), dim=1)
        if isinstance(state_proxy, list):
            states = torch.tensor(state_proxy, device=self.device)
        else:
            states = state_proxy

        dataset = {"states": states, "energies": scores}

        stats = self.get_statistics(scores)
        if self.normalize_data:
            dataset["energies"] = self.normalize(dataset["energies"], stats)

        return dataset, stats

    def get_statistics(self, y):
        """
        called each time the dataset is updated so has the most recent metrics
        """
        dict = {}
        dict["mean"] = torch.mean(y)
        dict["std"] = torch.std(y)
        dict["max"] = torch.max(y)
        dict["min"] = torch.min(y)
        return dict

    def normalize(self, y, stats):
        """
        Args:
            y: targets to normalize (tensor)
            mean: mean of targets (tensor)
            std: std of targets (tensor)
        Returns:
            y: normalized targets (tensor)
        """
        y = (y - stats["min"]) / (stats["max"] - stats["min"])
        # y = (y - stats["mean"]) / stats["std"]
        return y

    def denormalize(self, y, stats):
        """
        Args:
            y: targets to denormalize (tensor)
            mean: mean of targets (tensor)
            std: std of targets (tensor)
        Returns:
            y: denormalized targets (tensor)
        """
        y = y * (stats["max"] - stats["min"]) + stats["min"]
        # y = y * stats["std"] + stats["mean"]
        return y

    def update_dataset(self, states, energies, fidelity=None):
        """
        Args:
            queries: list of queries [[0, 0], [1, 1], ...]
            energies: list of energies [-0.6, -0.1, ...]
        Update the dataset with new data after AL iteration
        Updates the dataset stats
        Saves the updated dataset if save_data=True
        """

        energies = torch.tensor(energies, dtype=self.float, device=self.device)
        # fidelity = [state[-1] for state in states]
        get_figure_plots(
            self.env,
            states,
            energies,
            fidelity.squeeze(-1).tolist() if fidelity is not None else None,
            logger=self.logger,
            title="Sampled Dataset",
            key="post_al_iter_sampled_dataset",
            use_context=True,
        )

        samples = [self.env.state2readable(state) for state in states]
        readable_dataset = {
            "samples": samples,
            "energies": energies.tolist(),
        }
        self.logger.save_dataset(readable_dataset, "sampled")

        energies = self.scale_by_target_factor(energies)
        states_proxy = self.env.statebatch2proxy(states)

        train_energies, test_energies = [], []
        train_states, test_states = [], []
        train_samples, test_samples = [], []
        train_states_proxy, test_states_proxy = [], []
        for sample, state, state_proxy, energy in zip(
            samples, states, states_proxy, energies
        ):
            if np.random.uniform() < (1 / 10):
                test_samples.append(sample)
                test_states.append(state)
                test_states_proxy.append(state_proxy)
                test_energies.append(energy.item())
            else:
                train_samples.append(sample)
                train_states.append(state)
                train_states_proxy.append(state_proxy)
                train_energies.append(energy.item())

        test_states_proxy = self.convert_to_tensor(test_states_proxy)
        train_states_proxy = self.convert_to_tensor(train_states_proxy)
        test_energies = self.convert_to_tensor(test_energies)
        train_energies = self.convert_to_tensor(train_energies)

        if self.normalize_data:
            self.train_dataset["energies"] = self.denormalize(
                self.train_dataset["energies"], stats=self.train_stats
            )
            if self.test_dataset is not None:
                self.test_dataset["energies"] = self.denormalize(
                    self.test_dataset["energies"], stats=self.test_stats
                )

        self.train_dataset["energies"] = torch.cat(
            (self.train_dataset["energies"], train_energies), dim=0
        )
        if self.test_dataset is not None:
            self.test_dataset["energies"] = torch.cat(
                (self.test_dataset["energies"], test_energies), dim=0
            )

        self.train_dataset["states"] = torch.cat(
            (self.train_dataset["states"], train_states_proxy), dim=0
        )
        if self.test_dataset is not None:
            self.test_dataset["states"] = torch.cat(
                (self.test_dataset["states"], test_states_proxy), dim=0
            )

        self.train_stats = self.get_statistics(self.train_dataset["energies"])
        if self.test_dataset is not None:
            self.test_stats = self.get_statistics(self.test_dataset["energies"])
        if self.normalize_data:
            self.train_dataset["energies"] = self.normalize(
                self.train_dataset["energies"], self.train_stats
            )
            if self.test_dataset is not None:
                self.test_dataset["energies"] = self.normalize(
                    self.test_dataset["energies"], self.test_stats
                )
        self.train_data = Data(
            self.train_dataset["states"], self.train_dataset["energies"]
        )
        if self.test_dataset is not None:
            self.test_data = Data(
                self.test_dataset["states"], self.test_dataset["energies"]
            )

        self.logger.log_dataset_stats(self.train_stats, self.test_stats)
        if self.progress:
            print("\nUpdated Dataset Statistics")
            print(
                "\n Train \n \t Mean Score:{:.2f} \n \t  Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.train_stats["mean"],
                    self.train_stats["std"],
                    self.train_stats["min"],
                    self.train_stats["max"],
                )
            )
            if self.test_stats is not None:
                print(
                    "\n Test \n \t Mean Score:{:.2f}  \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                        self.test_stats["mean"],
                        self.test_stats["std"],
                        self.test_stats["min"],
                        self.test_stats["max"],
                    )
                )

        train_path = self.logger.data_path.parent / Path("data_train.csv")
        train_dataset = pd.read_csv(train_path, index_col=0)
        train_energies = self.scale_by_target_factor(train_energies)
        readable_train_dataset = {
            "samples": train_samples,
            "energies": train_energies.tolist(),
        }
        train_dataset = pd.concat([train_dataset, pd.DataFrame(readable_train_dataset)])
        self.logger.save_dataset(train_dataset, "train")

        if self.test_dataset is not None:
            test_path = self.logger.data_path.parent / Path("data_test.csv")
            test_dataset = pd.read_csv(test_path, index_col=0)
            test_energies = self.scale_by_target_factor(test_energies)
            readable_test_dataset = {
                "samples": test_samples,
                "energies": test_energies.tolist(),
            }
            test_dataset = pd.concat(
                [test_dataset, pd.DataFrame(readable_test_dataset)]
            )
            self.logger.save_dataset(test_dataset, "test")

    def collate_batch(self, batch):
        """
        Pads till maximum length in the batch
        """
        y, x = (
            [],
            [],
        )
        for (_sequence, _label) in batch:
            y.append(_label)
            x.append(_sequence)
        y = torch.tensor(y, dtype=self.float)  # , device=self.device
        xPadded = pad_sequence(x, batch_first=True, padding_value=0.0)
        return xPadded, y

    def get_dataloader(self):
        """
        Build and return the dataloader for the networks
        The dataloader should return x and y such that:
            x: self.env.statebatch2proxy(input)
            y: normalized (if need be) energies
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.dataloader.train.batch_size,
            shuffle=self.dataloader.train.shuffle,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_batch,
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=self.dataloader.test.batch_size,
            shuffle=self.dataloader.test.shuffle,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_batch,
        )

        return train_loader, test_loader
