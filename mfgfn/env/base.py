import pickle
import time

import torch
from gflownet.envs.base import GFlowNetEnv as BaseGFlowNetEnv


class GFlowNetEnv(BaseGFlowNetEnv):
    def __init__(
        self, proxy_state_format, beta_factor=0, norm_factor=1, proxy=None, **kwargs
    ):
        super().__init__(proxy=proxy, **kwargs)
        self.proxy_state_format = proxy_state_format
        self.beta_factor = beta_factor
        self.norm_factor = norm_factor
        if proxy is not None:
            self.set_proxy(proxy)

    def set_proxy(self, proxy):
        self.proxy = proxy
        if hasattr(self, "proxy_factor"):
            return
        if self.proxy is not None and self.proxy.maximize is not None:
            # can be None for dropout regressor/UCB
            maximize = self.proxy.maximize
        elif self.oracle is not None:
            maximize = self.oracle.maximize
        else:
            raise ValueError("Proxy and Oracle cannot be None together.")
        if maximize:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0

    # def get_test_batch(self, gflownet, test_pkl):

    #     times = {
    #         "all": 0.0,
    #         "forward_actions": 0.0,
    #         "backward_actions": 0.0,
    #         "actions_envs": 0.0,
    #         "rewards": 0.0,
    #     }
    #     t0_all = time.time()

    #     with open(test_pkl, "rb") as f:
    #         dict_tr = pickle.load(f)
    #         x_tr = dict_tr["x"]
    #     n_samples = len(x_tr)
    #     envs = [self.copy().reset(idx) for idx in range(n_samples)]
    #     envs_test = []
    #     actions = []
    #     valids = []
    #     for idx in range(len(x_tr)):
    #         env = envs[idx]
    #         env.n_actions = env.get_max_traj_len()
    #         # Required for env.done check in the mfenv env
    #         env = env.set_state(x_tr[idx], done=True)
    #         envs_test.append(env)
    #         actions.append((env.eos,))
    #         valids.append(True)

    #     while envs_test:
    #         # Sample backward actions
    #         with torch.no_grad():
    #             actions = gflownet.sample_actions(
    #                 envs_test,
    #                 times,
    #                 sampling_method="policy",
    #                 model=gflownet.backward_policy,
    #                 is_forward=False,
    #                 temperature=gflownet.temperature_logits,
    #                 random_action_prob=gflownet.random_action_prob,
    #             )
    #         # Add to batch
    #         batch = gflownet.sample_batch._add_to_batch(batch, envs_test, actions, valids)
    #         # Update environments with sampled actions
    #         envs_offline, actions, valids = self.step(
    #             envs_offline, actions, is_forward=False
    #         )
    #         assert all(valids)
    #         # Filter out finished trajectories
    #         if isinstance(env.state, list):
    #             envs_offline = [env for env in envs_offline if env.state != env.source]
    #         elif isinstance(env.state, TensorType):
    #             envs_offline = [
    #                 env
    #                 for env in envs_offline
    #                 if not torch.eq(env.state, env.source).all()
    #             ]
    #     return batch
