from typing import Any
import sys

sys.path.append(".")
import yaml
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import pocartpole
import optuna
import torch.nn.functional as F
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    NormalizeReward,
    NormalizeObservation,
)
from rl.algorithms.rppo import RPPO
from rl.common.tuning import TuningExperiment
from rl.models.simple_actor_critic import (
    SharedRecurrentAgent,
    RecurrentAgent,
    SimpleRecurrentAgent,
    SplittedRecurrentAgent,
)
from rl.common.config import Config, DictConfig, ExperimentConfig

if __name__ == "__main__":
    seed = 42

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    config = ExperimentConfig("./examples/tuning/exp_config.yml")
    build_env = lambda: gym.vector.make(
        "POCartPole-v1", asynchronous=False, num_envs=config.n_envs
    )
    build_test_env = lambda: gym.vector.make(
        "POCartPole-v1", asynchronous=False, num_envs=1, max_episode_steps=500
    )
    tuning = TuningExperiment(
        build_env, build_test_env, RPPO, SplittedRecurrentAgent, config
    )
    study = tuning.run(n_trials=100, n_jobs=-1)
    best_params = study.best_params
    print(f"Best params found: {best_params}")
    df = study.trials_dataframe(multi_index=True)
    df.to_csv("./examples/tuning/study.csv")
