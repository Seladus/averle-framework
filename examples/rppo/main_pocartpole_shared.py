import sys

sys.path.append(".")
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    NormalizeReward,
    NormalizeObservation,
)
from rl.algorithms.rppo import RPPO
from rl.models.simple_actor_critic import SimpleRecurrentAgent, SharedRecurrentAgent
from rl.common.config import Config
import pocartpole


if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    config = Config("./examples/rppo/pocartpole_config_shared.yml")
    env = gym.vector.make("POCartPole-v1", asynchronous=True, num_envs=config.n_envs)
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
    env.reset(seed=seed)
    agent = SharedRecurrentAgent(
        obs_dim, action_dim, f_hidden=[], f_enc=[64], f_actor=[64], f_critic=[64]
    )
    algo = RPPO(agent, config)
    algo.train(env, test_env)
