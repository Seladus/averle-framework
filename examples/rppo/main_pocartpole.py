from typing import Any
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
from rl.common.schedulers import PolynomialSchedule
from rl.common.config import Config
from rl.algorithms.rppo import RPPO
from rl.models.actor_critic import RecurrentAgent


if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    config = Config("./examples/rppo/pocartpole_config.yml")
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
    # env = NormalizeReward(env)
    env.reset(seed=seed)
    agent = RecurrentAgent(obs_dim, action_dim)
    algo = RPPO(agent, config)
    algo.train(env, test_env)
