from typing import Any
import yaml
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import pocartpole
import torch.nn.functional as F
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    NormalizeReward,
    NormalizeObservation,
)
from rl.algorithms.rppo import RPPO
from rl.models.simple_actor_critic import SimpleRecurrentAgent, SharedRecurrentAgent
from rl.common.config import Config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    config = Config("./config.yml")
    env = gym.vector.make("POCartPole-v1", asynchronous=False, num_envs=config.n_envs)
    test_env = gym.vector.make(
        "POCartPole-v1", asynchronous=False, num_envs=1, max_episode_steps=500
    )
    test_env = NormalizeObservation(test_env)
    obs_dim, action_dim = (
        env.unwrapped.single_observation_space.shape[-1],
        env.unwrapped.single_action_space.n,
    )
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    env.reset(seed=seed)
    # agent = SimpleRecurrentAgent(obs_dim, action_dim)
    agent = SharedRecurrentAgent(obs_dim, action_dim)

    algo = RPPO(agent, config)
    algo.train(env, test_env)
