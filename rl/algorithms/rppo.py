import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from rl.common.buffers.ppo_rollout_buffer import RecurrentRolloutBuffer
from collections import deque
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from rl.common.schedulers import PolynomialSchedule
from time import time
from rl.algorithms import RecurrentAlgorithm


class RPPO(RecurrentAlgorithm):
    def __init__(self, agent, config) -> None:
        super().__init__(agent, config)
        self.nb_epochs = config.nb_epochs if config.nb_epochs else 128
        self.epoch_steps = config.epoch_steps if config.epoch_steps else 256
        self.seq_len = config.seq_len if config.seq_len else 8
        self.batch_size = config.batch_size if config.batch_size else 64
        self.lr = float(config.lr) if config.lr else 1e-3
        self.target_lr = float(config.target_lr) if config.target_lr else None
        self.kl_limit = float(config.kl_limit) if config.kl_limit else None
        self.lr_scheduler = None
        self.nb_optim = config.nb_optim
        self.gamma = float(config.gamma) if config.gamma else 0.99
        self.gae_lambda = float(config.gae_lambda) if config.gae_lambda else 0.95
        self.clip_eps = float(config.clip_eps) if config.clip_eps else 0.2
        self.target_clip = (
            float(config.target_clip) if config.target_clip else self.clip_eps
        )
        self.clip_scheduler = PolynomialSchedule(
            self.clip_eps, self.target_clip, self.nb_epochs, 2.0
        )
        self.norm_adv = bool(config.norm_adv) if config.norm_adv else True
        self.max_grad_norm = float(config.max_grad_norm) if config.target_clip else None
        self.v_coef = float(config.v_coef) if config.v_coef else 0.5
        self.entropy_coef = float(config.entropy_coef) if config.entropy_coef else 0.0

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)
        self.lr_scheduler = (
            LambdaLR(
                self.optimizer,
                lambda epoch: PolynomialSchedule(
                    self.lr, self.target_lr, self.nb_epochs, 2.0
                )(epoch)
                / self.lr,
            )
            if self.target_lr is not None
            else None
        )

        self.update_steps = 0
        self.steps = 0
        self.current_lr = self.lr
        self.current_clip = self.clip_eps

    def _update_schedulers(self, epoch):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.current_lr = self.optimizer.param_groups[0]["lr"]
        self.clip_eps = self.clip_scheduler(epoch)

    def act(self, state, hidden, terminal=None):
        """used for inference"""
        self.agent.eval()
        n_envs = state.shape[0] if len(state.shape) > 1 else 1
        terminal = torch.Tensor(terminal).float().to(self.device)
        with torch.no_grad():
            probs, new_hidden = self.agent.actor(
                torch.from_numpy(state).float().view(1, n_envs, -1).to(self.device),
                hidden,
                terminal,
            )
            action = torch.argmax(probs, dim=-1)
        self.agent.train()
        return action.flatten().cpu().numpy(), new_hidden

    def train(self, env, test_env, nb_test_episodes=5, save=True, verbose=True):
        self.n_envs = env.num_envs
        best_reward = -np.inf
        super().train(env, test_env, save=save)
        episodic_rewards_queue = deque([], maxlen=100)
        test_episodic_rewards_queue = deque([], maxlen=20)
        for e in range(self.nb_epochs):
            t = time()
            obsv, _ = env.reset()
            buffer = RecurrentRolloutBuffer(self.device)
            terminal = torch.ones(self.n_envs)

            with torch.no_grad():
                hidden = self.agent.get_init_state(self.n_envs, self.device)

                rewards = np.zeros(self.n_envs)
                episodic_rewards = []

                for _ in range(self.epoch_steps):
                    old_hidden = hidden

                    # chose next action
                    state = torch.Tensor(obsv).float()
                    value, hidden = self.agent.critic(
                        state.view(1, self.n_envs, -1).to(self.device),
                        hidden,
                        terminal.to(self.device),
                    )
                    probs, hidden = self.agent.actor(
                        state.view(1, self.n_envs, -1).to(self.device),
                        hidden,
                        terminal.to(self.device),
                    )
                    action_dist = Categorical(probs)
                    action = action_dist.sample().flatten()
                    logprob = action_dist.log_prob(action).cpu()

                    # step environment
                    obsv, reward, done, truncated, infos = env.step(
                        action.cpu().numpy()
                    )
                    done = np.logical_or(done, truncated)
                    terminal = torch.Tensor(done).float()
                    rewards += reward

                    if done.any():
                        (ended_idxs,) = np.where(done)
                        episodic_rewards += [r for r in rewards[ended_idxs]]
                        rewards[ended_idxs] = 0

                    buffer.add(
                        old_hidden,
                        state,
                        value.flatten(),
                        action.flatten(),
                        logprob.flatten(),
                        torch.Tensor(reward).float(),
                        torch.Tensor(done).float(),
                    )
                state = torch.Tensor(obsv).float()
                final_value, _ = self.agent.critic(
                    state.view(1, self.n_envs, -1).to(self.device),
                    hidden,
                    terminal.to(self.device),
                )
                buffer.finish_rollout(
                    final_value.flatten(), done, self.gamma, self.gae_lambda
                )

                self.steps += self.epoch_steps

                if len(episodic_rewards) == 0:
                    episodic_rewards += [r for r in rewards]
                mean_rewards = np.mean(episodic_rewards)
                episodic_rewards_queue.append(mean_rewards)

                self.logger.add_scalar(
                    "train/episodic_returns",
                    mean_rewards,
                    self.steps,
                )
                self.logger.add_scalar(
                    "train/mean_episodic_returns",
                    np.mean(episodic_rewards_queue),
                    self.steps,
                )
                self.logger.add_scalar(
                    "execution/fps", self.nb_epochs / (time() - t), self.steps
                )

            (
                states,
                actions,
                logprobs,
                advantages,
                returns,
                actor_hidden,
                actor_cell,
                critic_hidden,
                critic_cell,
                masks,
            ), nb_seq = buffer.build_sequences(self.seq_len)

            clipfracs = []
            approx_kls = []
            pg_losses = []
            v_losses = []
            entropy_losses = []
            explained_vars = []

            t = time()
            # sample batchs and update models
            idxs = np.arange(nb_seq)
            for i in range(self.nb_optim):
                np.random.shuffle(idxs)
                for start in range(0, nb_seq, self.batch_size):
                    end = start + self.batch_size
                    b_idxs = idxs[start:end]

                    b_states = states[:, b_idxs]
                    b_actions = actions[:, b_idxs]
                    b_logprobs = logprobs[:, b_idxs]
                    b_advantages = advantages[:, b_idxs]
                    b_returns = returns[:, b_idxs]
                    b_actor_hidden = actor_hidden[:, b_idxs][0].permute(1, 0, 2)
                    b_actor_cell = actor_cell[:, b_idxs][0].permute(1, 0, 2)
                    b_critic_hidden = critic_hidden[:, b_idxs][0].permute(1, 0, 2)
                    b_critic_cell = critic_cell[:, b_idxs][0].permute(1, 0, 2)
                    b_masks = masks[b_idxs].T

                    # normalize advantages
                    if self.norm_adv:
                        adv_mean, adv_std = (
                            b_advantages.mean(),
                            b_advantages.std() + 1e-8,
                        )
                        b_advantages = (b_advantages - adv_mean) / adv_std

                    # setting hidden recurerent states
                    actor_hidden_states = (b_actor_hidden, b_actor_cell)
                    critic_hidden_states = (b_critic_hidden, b_critic_cell)
                    hidden_states = (actor_hidden_states, critic_hidden_states)

                    # compute policy gradient loss
                    probs, _ = self.agent.actor(b_states, hidden_states)
                    action_dist = Categorical(probs)
                    new_logprobs = action_dist.log_prob(b_actions)

                    logratio = new_logprobs - b_logprobs
                    prob_ratio = torch.exp(logratio)

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((prob_ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((prob_ratio - 1.0).abs() > self.clip_eps)
                            .cpu()
                            .float()
                            .mean()
                            .item()
                        ]

                    # compute entropy loss
                    entropy = action_dist.entropy() * b_masks
                    entropy_loss = -torch.mean(entropy)

                    # compute policy loss
                    pg_loss1 = prob_ratio * b_advantages
                    pg_loss2 = (
                        torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * b_advantages
                    )
                    pg_loss = torch.min(pg_loss1, pg_loss2) * b_masks
                    pg_loss = -torch.mean(pg_loss)

                    # compute critic loss
                    values, _ = self.agent.critic(b_states, hidden_states)
                    v_loss = F.mse_loss(
                        b_returns * b_masks, values.squeeze(-1) * b_masks
                    )

                    # compute final loss
                    loss = (
                        pg_loss
                        + self.v_coef * v_loss
                        + self.entropy_coef * entropy_loss
                    )

                    if self.kl_limit is None or self.kl_limit > approx_kl:
                        self.optimizer.zero_grad()
                        loss.backward()
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                self.agent.parameters(), self.max_grad_norm
                            )
                        self.optimizer.step()

                    y_pred, y_true = (
                        values.detach().view(self.seq_len, -1).cpu().numpy(),
                        b_returns.detach().cpu().numpy(),
                    )
                    var_y = np.var(y_true)
                    explained_var = (
                        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                    )

                    # add metrics to update history
                    approx_kls.append(approx_kl.cpu().item())
                    pg_losses.append(pg_loss.cpu().item())
                    v_losses.append(v_loss.cpu().item())
                    entropy_losses.append(entropy.mean().cpu().item())
                    explained_vars.append(explained_var)

                self.update_steps += 1

            self.logger.add_scalar(
                "execution/ups", self.nb_optim / (time() - t), self.steps
            )
            self.logger.add_scalar("train/lr", self.current_lr, self.steps)
            self.logger.add_scalar("train/clip", self.clip_eps, self.steps)
            self.logger.add_scalar("train/value_loss", np.mean(v_losses), self.steps)
            self.logger.add_scalar("train/policy_loss", np.mean(pg_losses), self.steps)
            self.logger.add_scalar("train/entropy", np.mean(entropy_losses), self.steps)
            self.logger.add_scalar("train/approx_kl", np.mean(approx_kls), self.steps)
            self.logger.add_scalar("train/clipfrac", np.mean(clipfracs), self.steps)
            self.logger.add_scalar(
                "train/explained_variance", np.mean(explained_vars), self.steps
            )

            # evaluation
            eval = self.evaluate(test_env, nb_test_episodes, seed=self.test_seed)
            test_episodic_rewards_queue.append(eval)
            self.logger.add_scalar("test/episodic_returns", eval, self.steps)
            self.logger.add_scalar(
                "test/avg_episodic_returns",
                np.mean(test_episodic_rewards_queue),
                self.steps,
            )
            # save best model
            if eval >= best_reward:
                if save:
                    self.save(f"best_model_{eval:.2f}.pt")
                best_reward = eval

            # save model
            # self.save(f"{self.algo}_{self.steps}_{eval:.1f}.pt")

            # update learning rate
            self._update_schedulers(e)

            if verbose:
                print(
                    f"EPOCH {e} - mean reward : {mean_rewards} - eval reward : {eval}"
                )

        end_metrics = {
            "avg_episodic_returns": np.mean(test_episodic_rewards_queue),
            "best_reward": best_reward,
        }

        self.logger.add_hparams(self.hparams, end_metrics)

        return end_metrics
