import torch
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
import os


class ProxyBotorchUCB(Model):
    def __init__(self, regressor, num_dropout_samples):
        super().__init__()
        self.regressor = regressor
        self._num_outputs = 1
        self.num_dropout_samples = num_dropout_samples

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        self.regressor.model.train()
        dim = X.ndim
        if dim == 3:
            X = X.squeeze(-2)

        with torch.no_grad():
            outputs = self.regressor.forward_with_uncertainty(
                X, self.num_dropout_samples
            )
        mean = torch.mean(outputs, dim=1).unsqueeze(-1)
        var = torch.var(outputs, dim=1).unsqueeze(-1)
        # if var is an array of zeros then we add a small value to it
        var = torch.where(var == 0, torch.ones_like(var) * 1e-4, var)

        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis=0)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])


class MultifidelityOracleModel(Model):
    def __init__(self, oracle, n_fid, device):
        super().__init__()
        self.oracle = oracle
        self._num_outputs = 1
        self.n_fid = n_fid
        self.device = device

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        """
        In this implemenntation, the mean is the same irrespective of the fidelity.
        Just the cost and the uncertainty depend on the fidelity.
        The mean is the true score, so mean_curr_fidelity = mean_max_fidelity.
        """
        if len(X.shape) == 2:
            fid_tensor = X[:, -1]
            state_tensor = X[:, :-1]
            scores = self.oracle[self.n_fid - 1](state_tensor)
            var = torch.zeros(X.shape[0])
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                var[idx_fid] = self.oracle[fid].sigma ** 2
            # candidate set
            # if torch.all(torch.eq(fid_tensor, self.n_fid - 1)):
            #     # posterior max samples generated from candidate set
            #     var = torch.ones((X.shape[0])) * self.oracle[0].sigma ** 2
            #     # check shape of scores
            # else:
            #     # posterior of candidate set
            #     # mean = torch.zeros(X.shape[0])
            #     var = torch.zeros(X.shape[0])
            #     for fid in range(self.n_fid):
            #         idx_fid = torch.where(fid_tensor == fid)[0]
            #         var[idx_fid] = self.oracle[fid].sigma ** 2
            mean = scores
            covar = torch.diag(var)
        # batch and projected batch
        elif len(X.shape) == 4:
            # combination of target and original fidelity test set
            #  X = 32 x 1 x 2 x 3
            var_curr_fidelity = torch.zeros(X.shape[0], 1)
            states = X[:, :, 0, :-1].squeeze(-2)
            states = states.squeeze(-2)
            scores = self.oracle[self.n_fid - 1](states)
            scores = scores.unsqueeze(-1)
            # mean_curr_fidelity = torch.zeros(X.shape[0], 1)
            fid_tensor = X[:, :, 0, -1]
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                var_curr_fidelity[idx_fid] = self.oracle[fid].sigma ** 2
            # mean = torch.stack((mean_curr_fidelity, mean_max_fidelity), dim=2)
            mean = torch.stack((scores, scores), dim=2)
            var_max_fidelity = self.oracle[self.n_fid - 1].sigma ** 2
            covar = [
                torch.diag(torch.FloatTensor([var_curr_fidelity[i], var_max_fidelity]))
                for i in range(X.shape[0])
            ]
            covar = torch.stack(covar, axis=0)
            covar = covar.unsqueeze(1)
        # batch
        elif len(X.shape) == 3:
            fid_tensor = X[:, :, -1]
            state_tensor = X[:, :, :-1]
            state_tensor = state_tensor.squeeze(-2)
            scores = self.oracle[self.n_fid - 1](state_tensor)
            mean = scores.unsqueeze(-1)
            # mean = torch.ones((X.shape[0], 1))
            var = torch.ones((X.shape[0], 1))
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                # state_fid = state_tensor[idx_fid]
                var[idx_fid] = self.oracle[fid].sigma ** 2
            # posterior of max samples
            # var = torch.ones((X.shape[0], 1)) * self.oracle[0].sigma ** 2
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)

        mean = mean.to(self.device)
        covar = covar.to(self.device)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior
