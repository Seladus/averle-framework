from collections import deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from rl.common.buffers.replay_buffer import Batch, ReplayBuffer
from rl.common.config import Config
from rl.common.schedulers import PolynomialSchedule
from time import time
from rl.algorithms import Algorithm
from rl.models import QAgent


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DQN(Algorithm):
    def __init__(self, agent: QAgent, config: Config) -> None:
        super().__init__(agent, config)
        self.nb_epochs = config.nb_epochs if config.nb_epochs else 128
        self.epoch_steps = config.epoch_steps if config.epoch_steps else 256
        self.buffer_size = int(config.buffer_size) if config.buffer_size else int(2e6)
        self.batch_size = int(config.batch_size) if config.batch_size else 64
        self.warmup_steps = (
            int(config.warmup_steps) if config.warmup_steps else self.buffer_size // 2
        )
        self.gamma = config.gamma if config.gamma else 0.99
        self.tau = config.tau if config.tau else 1e-3
        self.target_update = config.target_update if config.target_update else "soft"
        self.target_update_freq = (
            int(config.target_update_freq) if config.target_update_freq else 1
        )
        self.update_freq = config.update_freq if config.update_freq else 1
        self.eps = config.eps if config.eps else 1.0
        self.eps_final = config.eps_final if config.eps_final else 0.2
        self.eps_decrease_steps = (
            config.eps_decrease_steps if config.eps_decrease_steps else int(5e6)
        )
        self.eps_scheduler = PolynomialSchedule(
            self.eps, self.eps_final, self.eps_decrease_steps, 2.0
        )
        self.lr = float(config.lr) if config.lr else 1e-3

        self.target = copy.deepcopy(self.agent).to(self.device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)

        self.update_steps = 0
        self.steps = 0

    def warmup(self, env, replay_buffer):
        next_obs, _ = env.reset()
        for i in range(self.warmup_steps):
            obs = next_obs
            action = env.action_space.sample()
            next_obs, reward, done, truncated, infos = env.step(action)
            self.replay_buffer.add(obs, action, next_obs, reward, done, truncated)

    def train(
        self,
        env: gym.vector.VectorEnv,
        test_env: gym.vector.VectorEnv,
        nb_test_episodes=5,
        save=True,
        verbose=True,
    ):
        obs_shape = env.single_observation_space.shape
        self.n_envs = env.num_envs
        super().train(env, test_env, save=save)

        self.replay_buffer = ReplayBuffer(
            obs_shape,
            self.buffer_size,
            self.n_envs,
        )
        self.warmup(env, self.replay_buffer)
        for e in range(self.nb_epochs):
            next_obs, _ = env.reset()
            for _ in range(self.epoch_steps):
                obs = next_obs
                if np.random.random() < self.eps_scheduler(self.steps):
                    action = env.action_space.sample()
                else:
                    action = (
                        self.agent.q(torch.from_numpy(obs).to(self.device))
                        .argmax(dim=-1)
                        .cpu()
                        .numpy()
                    )
                next_obs, reward, done, truncated, infos = env.step(action)
                self.replay_buffer.add(obs, action, next_obs, reward, done, truncated)

                batch = self.replay_buffer.sample(self.batch_size)

                if self.steps % self.update_freq == 0:
                    self.update(batch)

                self.steps += 1

            eval = self.evaluate(test_env, nb_test_episodes, seed=self.test_seed)
            self.logger.add_scalar("test/episodic_returns", eval, self.steps)
            self.logger.add_scalar(
                "train/espilon", self.eps_scheduler(self.steps), self.steps
            )

        batch: Batch = self.replay_buffer.sample(self.batch_size)

    def update(self, batch: Batch):
        obs = torch.from_numpy(batch.obs).float().to(self.device)
        next_obs = torch.from_numpy(batch.next_obs).float().to(self.device)
        done = torch.from_numpy(batch.done).float().to(self.device)
        rewards = torch.from_numpy(batch.reward).float().to(self.device)
        action = torch.from_numpy(batch.action).long().to(self.device)
        b_size = obs.shape[0]
        q_tm1 = self.agent.q(obs)[torch.arange(b_size), action]
        q_t = self.agent.q(next_obs)
        q_target_t = self.target.q(next_obs)
        target_tm1 = (
            rewards
            + self.gamma
            * (1 - done)
            * q_target_t[torch.arange(b_size), q_t.argmax(dim=-1)]
        ).detach()

        loss = F.mse_loss(target_tm1, q_tm1)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        self.logger.add_scalar("train/q", q_t.mean(), self.update_steps)
        self.logger.add_scalar("train/q_target", q_target_t.mean(), self.update_steps)
        self.logger.add_scalar("train/loss", loss, self.update_steps)

        self.update_steps += 1

        if self.update_steps % self.target_update_freq == 0:
            if self.target_update == "soft":
                soft_update(self.target, self.agent, self.tau)
            else:
                hard_update(self.target, self.agent)

    def act(self, state):
        self.agent.eval()
        n_envs = state.shape[0] if len(state.shape) > 1 else 1
        with torch.no_grad():
            q = self.agent.q(torch.from_numpy(state).view(n_envs, -1).to(self.device))
            action = q.cpu().argmax(dim=-1)
        self.agent.train()
        return action.numpy()
