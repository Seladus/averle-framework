from typing import Any
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
from rl.common.config import DictConfig


if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    env = gym.vector.make("POCartPole-v1", asynchronous=False, num_envs=1)
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

    config = DictConfig()
    algo = RPPO("saves/RPPO_20736_500.000000.pt", config)
    eval = algo.evaluate(test_env, 1)
    print(f"Average Reward: {eval}")
