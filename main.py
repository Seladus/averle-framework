import yaml
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import pocartpole
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    NormalizeReward,
    NormalizeObservation,
)
from rl.algorithms.rppo import RPPO
from rl.models.actor_critic import Actor, Critic


class RecurrentAgent(nn.Module):
    def __init__(self, state_dim, action_dim, h=64, recurrent_layers=1) -> None:
        super().__init__()
        self.actor_net = Actor(state_dim, action_dim, h, recurrent_layers)
        self.critic_net = Critic(state_dim, h, recurrent_layers)

    def get_init_state(self, batch_size, device):
        return (
            self.actor_net.get_init_state(batch_size, device),
            self.critic_net.get_init_state(batch_size, device),
        )

    def critic(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        value_out, critic_hidden = self.critic_net(state, critic_hidden, terminal)
        return value_out, (actor_hidden, critic_hidden)

    def actor(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        action_dist, actor_hidden = self.actor_net(state, actor_hidden, terminal)
        return action_dist, (actor_hidden, critic_hidden)


class Config:
    def __init__(self, path="./config.yml") -> None:
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)
        for key, value in self.config.items():
            self.__setattr__(key, value)

    def dict(self):
        return self.config


if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    config = Config("./config.yml")
    env = gym.vector.make("POCartPole-v1", asynchronous=False, num_envs=config.n_envs)
    obs_dim, action_dim = (
        env.unwrapped.single_observation_space.shape[-1],
        env.unwrapped.single_action_space.n,
    )
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    # env = NormalizeReward(env)
    env.reset(seed=seed)
    agent = RecurrentAgent(obs_dim, action_dim)
    algo = RPPO(agent, config)
    algo.train(env)
