import numpy as np
import torch
from gflownet.proxy.base import Proxy


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, **kwargs) -> None:
        super().__init__(**kwargs)
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        if hasattr(self.regressor, "load_model") and not self.regressor.load_model():
            raise FileNotFoundError
        elif hasattr(self.regressor, "load_model") == False:
            print("Model has not been loaded from path.")

    def __call__(self, inputs):
        """
        Args:
            inputs: proxy-compatible input tensor
            dim = n_samples x obs_dim

        Returns:
            vanilla rewards
                - (with no power/boltzmann) transformation
                - dim = n_samples
                - ndim = 1

        """
        # TODO: Resolve: when called to get rewards, input is tensor
        # But when called to get scores after GFN training, input is numpy array
        self.regressor.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(self.device)
        self.regressor.model.train()
        with torch.no_grad():
            output = self.regressor.model(inputs).squeeze(-1)
        return output
