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
from rl.models.simple_actor_critic import (
    SharedRecurrentAgent,
    RecurrentAgent,
    SimpleRecurrentAgent,
    SplittedRecurrentAgent,
)
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
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    obs_dim, action_dim = (
        env.single_observation_space.shape[-1],
        env.single_action_space.n,
    )
    # env = NormalizeReward(env)
    env.reset(seed=seed)
    # agent = SimpleRecurrentAgent(obs_dim, action_dim)
    agent = SharedRecurrentAgent(
        obs_dim, action_dim, f_hidden=[], f_enc=[64], f_actor=[64], f_critic=[64]
    )
    # agent = SplittedRecurrentAgent(obs_dim, action_dim, f_actor_enc=[], f_critic_enc=[])
    print(agent)
    algo = RPPO(agent, config)
    algo.train(env, test_env)
