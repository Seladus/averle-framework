import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl.common.buffers.ppo_rollout_buffer import RecurrentRolloutBuffer
from collections import deque


class RPPO:
    def __init__(self, agent, config) -> None:
        self.device = config.device
        self.n_envs = config.n_envs
        self.nb_epochs = config.nb_epochs
        self.epoch_steps = config.epoch_steps
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.lr = float(config.lr)
        self.nb_optim = config.nb_optim
        self.gamma = float(config.gamma)
        self.gae_lambda = float(config.gae_lambda)
        self.clip_eps = float(config.clip_eps)
        self.norm_adv = bool(config.norm_adv)

        self.agent = agent

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)

        self.logger = SummaryWriter()

        self.update_steps = 0
        self.steps = 0

    def train(self, env):
        episodic_rewards_queue = deque([], maxlen=100)
        for i in range(self.nb_epochs):
            obsv, _ = env.reset()
            buffer = RecurrentRolloutBuffer()
            terminal = torch.ones(self.n_envs)

            with torch.no_grad():
                hidden = self.agent.get_init_state(self.n_envs, self.device)
                episodic_returns = []
                episodic_length = []

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
                    action_dist, hidden = self.agent.actor(
                        state.view(1, self.n_envs, -1).to(self.device),
                        hidden,
                        terminal.to(self.device),
                    )
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

                    if "episode" in infos:
                        (ended_idxs,) = np.where(infos["_episode"])
                        episodic_returns += [
                            r for r in infos["episode"]["r"][ended_idxs]
                        ]
                        episodic_length += [
                            l for l in infos["episode"]["l"][ended_idxs]
                        ]

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

                print(
                    f"EPOCH {i} - mean reward : {np.mean(episodic_returns)} - episode length : {np.mean(episodic_length)}"
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

            # sample batchs and update models
            idxs = np.arange(nb_seq)
            clipfracs = []
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
                    action_dist, _ = self.agent.actor(b_states, hidden_states)
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
                    loss = pg_loss + 0.5 * v_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                    self.optimizer.step()
                self.update_steps += 1

            y_pred, y_true = (
                values.detach().view(self.seq_len, -1).cpu().numpy(),
                b_returns.detach().cpu().numpy(),
            )
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            self.logger.add_scalar(
                "train/value_loss", v_loss.detach().cpu().item(), self.update_steps
            )
            self.logger.add_scalar(
                "train/policy_loss", pg_loss.detach().cpu().item(), self.update_steps
            )
            self.logger.add_scalar(
                "train/approx_kl", approx_kl.detach().cpu().item(), self.update_steps
            )
            self.logger.add_scalar(
                "train/clipfrac", np.mean(clipfracs), self.update_steps
            )
            self.logger.add_scalar(
                "train/explained_variance", explained_var, self.update_steps
            )
