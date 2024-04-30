import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from gflownet.envs.base import Buffer
from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.common import set_device, set_float_precision, torch2np
from torch.distributions import Bernoulli, Categorical
from torchtyping import TensorType
from tqdm import tqdm


class Policy:
    def __init__(self, config, env, device, float_precision):
        self.n_hid = config.n_hid
        self.n_layers = config.n_layers
        self.device = device
        self.float = float_precision
        self.model = self.make_mlp(
            [env.policy_input_dim]
            + [config.n_hid] * config.n_layers
            + [env.policy_output_dim + 1]
        )
        self.is_model = True
        if self.is_model:
            self.model.to(self.device)

    def make_mlp(self, l, act=nn.LeakyReLU(), tail=[]):
        """makes an MLP with no top layer activation"""
        return nn.Sequential(
            *(
                sum(
                    [
                        [nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
                        for n, (i, o) in enumerate(zip(l, l[1:]))
                    ],
                    [],
                )
                + tail
            )
        ).to(dtype=self.float)

    def random_distribution(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def __call__(self, states):
        return self.model(states)


class PPOAgent(GFlowNetAgent):
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        policy,
        ppo_num_epochs,
        ppo_epoch_size,
        logger,
        random_action_prob=0.0,
        ppo_clip=0.1,
        ppo_entropy_coef=1e-3,
        temperature_logits=1,
        active_learning=False,
        sample_only=False,
        *args,
        **kwargs,
    ):
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env = env
        self.mask_source = self._tbool([self.env.get_mask_invalid_actions_forward()])
        self.ppo_num_epochs = ppo_num_epochs
        self.clip_param = ppo_clip
        self.entropy_coef = ppo_entropy_coef
        self.ttsr = ppo_num_epochs
        self.sttr = ppo_epoch_size
        self.logger = logger
        self.n_train_steps = optimizer.n_train_steps
        self.batch_size = optimizer.batch_size
        self.ema_alpha = optimizer.ema_alpha
        self.early_stopping = optimizer.early_stopping
        self.forward_policy = Policy(policy, self.env, self.device, self.float)
        if "checkpoint" in policy:
            self.logger.set_forward_policy_ckpt_path(policy.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        self.opt, self.lr_scheduler = make_opt(
            self.forward_policy.model.parameters(), optimizer
        )
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.clip_grad_norm = optimizer.clip_grad_norm
        # self.pct_offline = pct_offline
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0

    def sample_actions(
        self,
        envs,
        sampling_method="policy",
        model=None,
        temperature=1.0,
    ):
        """
        Samples one action on each environment of a list.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        sampling_method : string
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if sampling_method="policy"

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """

        if sampling_method == "random":
            random_action_prob = 1.0
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]
        mask_invalid_actions = self._tbool(
            [env.get_mask_invalid_actions_forward() for env in envs]
        )
        policy_outputs = model(self._tfloat(self.env.statebatch2policy(states)))
        # Skip v from policy outputs
        policy_outputs = policy_outputs[:, :-1]
        # Sample actions from policy outputs
        actions, logprobs = self.env.sample_actions(
            policy_outputs,
            sampling_method,
            mask_invalid_actions,
            temperature,
        )  # actions is list of tuples
        return actions, logprobs

    def loss(self, it, batch, masks_s, G, advantages, loginf=1000):
        loginf = self._tfloat([loginf])

        (
            _,
            _,
            _,
            _,
            _,
            is_terminal,
            _,
            _,
        ) = zip(*batch)
        is_terminal = torch.cat(is_terminal, 0)

        all_rewards = torch.stack(G)
        ret_rewards = all_rewards[is_terminal.eq(1)]

        idxs = torch.randint(low=0, high=len(batch), size=(self.batch_size,))

        # Concatenate lists of tensors in idx
        (
            states,
            actions,
            logprobs,
            _,
            _,
            is_terminal,
            _,
            _,
        ) = [torch.cat(i, 0) for i in zip(*[batch[i] for i in idxs])]
        masks_s = torch.stack(masks_s)[idxs]
        G = torch.stack(G)[idxs]
        advantages = torch.stack(advantages)[idxs]

        policy_outputs = self.forward_policy(self.env.statetorch2policy(states))
        logits, values = policy_outputs[:, :-1], policy_outputs[:, -1]
        if masks_s is not None:
            logits[masks_s] = -loginf
        action_indices = (
            torch.tensor(
                [
                    self.env.action_space.index(tuple(action.tolist()))
                    for action in actions
                ]
            )
            .to(int)
            .to(self.device)
        )

        new_pol = Categorical(logits=logits)
        new_logprobs = new_pol.log_prob(action_indices)
        ratio = torch.exp(new_logprobs - logprobs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (G - values).pow(2).mean()
        entropy = new_pol.entropy().mean()
        loss = action_loss + value_loss - entropy * self.entropy_coef

        return (
            loss,
            action_loss,
            value_loss,
        ), ret_rewards

    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data

        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] s, the state
                - [1] a, the action
                - [2] lp, log_prob of action
                - [3] mask_s
                - [4] sp, the next state
                - [5] done [True, False]
                - [6] traj id: identifies each trajectory
                - [7] state id: identifies each state within a traj
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states (state)

        Args
        ----
        """

        def _add_to_trajectory(
            trajs, curr_states, actions, logprobs, envs_sp, valids, train=True
        ):
            for s, action, logprob, env_sp, valid in zip(
                curr_states, actions, logprobs, envs_sp, valids
            ):
                if valid is False:
                    continue
                parents, _ = env.get_parents(action=action)
                mask_sp = env_sp.get_mask_invalid_actions_forward()
                if train:
                    trajs[env_sp.id].append(
                        # batch.append(
                        [
                            self._tfloat([s]),
                            self._tfloat([action]),
                            self._tfloat([logprob]),
                            self._tfloat([env_sp.state]),
                            self._tbool([mask_sp]),
                            self._tbool([env_sp.done]),
                            self._tlong([env_sp.id]),
                            self._tlong([env_sp.n_actions]),
                        ]
                    )

                else:
                    # TODO: to fix sampling
                    if env_sp.done:
                        trajs[env_sp.id].append(env_sp.state)
            return trajs

        times = {
            "actions_envs": 0.0,
        }
        trajs = defaultdict(list)
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        elif n_samples is not None and n_samples > 0:
            envs = [self.env.copy().reset(idx) for idx in range(n_samples)]
        else:
            return None, None

        while envs:
            # Sample forward actions
            with torch.no_grad():
                if train is False:
                    actions, logprobs = self.sample_actions(
                        envs,
                        sampling_method="policy",
                        model=self.forward_policy,
                        temperature=1.0,
                    )
                else:
                    actions, logprobs = self.sample_actions(
                        envs,
                        sampling_method="policy",
                        model=self.forward_policy,
                        temperature=1.0,
                    )
            s = []
            for env in envs:
                if env.is_state_list == True:
                    s.append(env.state.copy())
                else:
                    s.append(env.state.clone())

            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, is_forward=True)
            # Add to batch
            t0_a_envs = time.time()
            trajs = _add_to_trajectory(trajs, s, actions, logprobs, envs, valids, train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
        if train is False:
            batch = sum(trajs.values(), [])
            return batch, times

        batch = sum(trajs.values(), [])
        states, _, _, states_sp, masks_sp, done, traj_id, state_id = [
            torch.cat(i, 0) for i in zip(*batch)
        ]

        masks_s = torch.cat(
            [
                (
                    masks_sp[torch.where((state_id == sid - 1) & (traj_id == pid))]
                    if sid > 1
                    else self.mask_source
                )
                for sid, pid in zip(state_id, traj_id)
            ]
        )
        rewards = self.env.reward_torchbatch(states, done)

        # G is simply reward of the trajectory
        G = rewards.clone()
        non_zero_indices = torch.nonzero(G).squeeze()
        non_zero_elements = G[non_zero_indices]
        for i in range(len(non_zero_indices)):
            if i == 0:
                G[0 : non_zero_indices[i]] = non_zero_elements[i]
            else:
                G[non_zero_indices[i - 1] + 1 : non_zero_indices[i]] = (
                    non_zero_elements[i]
                )
        # use all s and sp to compute all adv
        with torch.no_grad():
            v_s = self.forward_policy(self.env.statetorch2policy(states))[:, -1]
            v_sp = self.forward_policy(self.env.statetorch2policy(states_sp))[:, -1]
        adv = rewards + v_sp * torch.logical_not(done) - v_s
        return batch, masks_s, G, adv, times

    def unpack_terminal_states(self, batch):
        """
        Unpacks the terminating states and trajectories of a batch and converts them
        to Python lists/tuples.
        """
        trajs = [[] for _ in range(self.batch_size)]
        states = [None] * self.batch_size
        for el in batch:
            traj_id = el[6][:1].item()
            state_id = el[7][:1].item()
            trajs[traj_id].append(tuple(el[1][0].tolist()))
            if bool(el[5].item()):
                states[traj_id] = tuple(el[0][0].tolist())
        trajs = [tuple(el) for el in trajs]
        return states, trajs

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Generate list of environments
        envs = [self.env.copy().reset() for _ in range(self.batch_size)]
        # Train loop
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress)
        # PPOEdit: added for when first_iter = False for test
        x_sampled = None
        for it in pbar:
            # Test
            if self.logger.do_test(it):
                (
                    self.l1,
                    self.kl,
                    self.jsd,
                    self.corr,
                    x_sampled,
                    kde_pred,
                    kde_true,
                ) = self.test()
                self.logger.log_test_metrics(
                    self.l1, self.kl, self.jsd, self.corr, it, self.use_context
                )
            if self.logger.do_plot(it) and x_sampled is not None and len(x_sampled) > 0:
                figs = self.plot(x_sampled, kde_pred, kde_true)
                self.logger.log_plots(figs, it, self.use_context)
            t0_iter = time.time()
            data = []
            data_masks_s = []
            data_G = []
            data_adv = []
            # rollout phase
            for j in range(self.sttr):  # ppo_epoch_size
                batch, masks_s, G, adv, times = self.sample_batch(envs)
                data += batch
                data_masks_s += masks_s
                data_G += G
                data_adv += adv
            # data = tuple(data)
            # learning phase
            for j in range(self.ppo_num_epochs):
                losses, rewards = self.loss(
                    it * self.ttsr + j, data, data_masks_s, data_G, data_adv
                )  # returns (opt loss, *metrics)
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    if self.opt is not None:
                        # required for when fp is uniform
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Buffer
            states_term, trajs_term = self.unpack_terminal_states(batch)

            proxy_vals = self.env.reward2proxy(rewards).tolist()
            rewards = rewards.tolist()
            if self.logger.do_test(it) and hasattr(self.env, "get_cost"):
                costs = self.env.get_cost(states_term)
            else:
                costs = 0.0
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term
            else:
                all_visited.extend(states_term)
            # Progress bar
            self.logger.progressbar_update(
                pbar, all_losses, rewards, self.jsd, it, self.use_context
            )
            # Train logs
            t0_log = time.time()
            if hasattr(self.env, "unpad_function"):
                unpad_function = self.env.unpad_function
            else:
                unpad_function = None
            self.logger.log_train(
                losses=losses,
                rewards=rewards,
                proxy_vals=proxy_vals,
                states_term=states_term,
                costs=costs,
                batch_size=len(data),
                logz=None,
                learning_rates=self.lr_scheduler.get_last_lr(),
                step=it,
                use_context=self.use_context,
                unpad_function=unpad_function,
            )
            t1_log = time.time()
            times.update({"log": t1_log - t0_log})
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(self.forward_policy, None, step=it)
            t1_model = time.time()
            times.update({"save_interim_model": t1_model - t0_model})

            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1] + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2] + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]
            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

        # Save final model
        self.logger.save_models(self.forward_policy, None, final=True)
        # Close logger
        if self.use_context == False:
            self.logger.end()


def make_opt(params, config):
    """
    Set up the optimizer
    """
    params = list(params)
    if not len(params):
        return None
    if config.method == "adam":
        opt = torch.optim.Adam(
            params,
            config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
    elif config.method == "msgd":
        opt = torch.optim.SGD(params, config.lr, momentum=config.momentum)
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=config.lr_decay_period,
        gamma=config.lr_decay_gamma,
    )
    return opt, lr_scheduler
