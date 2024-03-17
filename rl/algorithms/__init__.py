import torch
import numpy as np
import os
import shutil
from torch import nn
from gymnasium.vector import VectorEnv
from rl.common.config import Config
from rl.models import RecurrentActorCriticAgent, ActorCriticAgent
from torch.utils.tensorboard import SummaryWriter


class Algorithm:
    def __init__(self, agent, config: Config) -> None:
        self.algo = self.__class__.__name__
        self.hparams = config.dict()

        self.test_seed = config.test_seed
        self.device = config.device if config.device else "cpu"
        self.n_envs = int(config.n_envs) if config.n_envs else 1

        if type(agent) == str:
            self.load(agent)
        else:
            self.agent = agent
        self.agent.to(self.device)

        self.save_folder = config.save_folder if config.save_folder else "saves"
        self.save_freq = int(config.save_freq) if config.save_freq else 1

        n = 1
        log_folder = config.log_folder if config.log_folder else "runs"
        self.log_folder = None
        while self.log_folder is None or os.path.exists(self.log_folder):
            self.log_folder = os.path.join(log_folder, f"{self.algo}_{n}")
            n += 1

    def act(self, state):
        raise NotImplementedError()

    def evaluate(self, env: VectorEnv, n_episodes, seed=None):
        n_envs = env.num_envs

        obs, _ = env.reset(seed=seed)
        rewards = np.zeros(n_envs)
        episodic_rewards = []
        while len(episodic_rewards) < n_episodes:
            action = self.act(obs)
            obs, reward, done, truncated, infos = env.step(action)
            done = np.logical_or(done, truncated)
            rewards += reward

            if done.any():
                (ended_idxs,) = np.where(done)
                episodic_rewards += [r for r in rewards[ended_idxs]]
                rewards[ended_idxs] = 0

        return np.mean(episodic_rewards)

    def train(self, env: VectorEnv, test_env: VectorEnv, save=True):
        if save:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.mkdir(self.save_folder)
        self.logger = SummaryWriter(self.log_folder)

    def save(self, name):
        torch.save(self.agent, os.path.join(self.save_folder, name))

    def load(self, path):
        self.agent = torch.load(path)


class RecurrentAlgorithm(ActorCriticAgent):
    def __init__(self, agent: RecurrentActorCriticAgent, config: Config) -> None:
        super().__init__(agent, config)

    def act(self, state, hidden, terminal=None):
        raise NotImplementedError()

    def evaluate(self, env: VectorEnv, n_episodes, seed=None):
        n_envs = env.num_envs

        obs, _ = env.reset(seed=seed)
        terminal = torch.ones(n_envs)
        hidden = self.agent.get_init_state(n_envs, self.device)
        rewards = np.zeros(n_envs)
        episodic_rewards = []
        while len(episodic_rewards) < n_episodes:
            action, hidden = self.act(obs, hidden, terminal)
            obs, reward, done, truncated, infos = env.step(action)
            done = np.logical_or(done, truncated)
            terminal = torch.Tensor(done).float()
            rewards += reward

            if done.any():
                (ended_idxs,) = np.where(done)
                episodic_rewards += [r for r in rewards[ended_idxs]]
                rewards[ended_idxs] = 0

        return np.mean(episodic_rewards)
