import numpy as np
import optuna
import torch
from rl.common.config import DictConfig, ExperimentConfig


class TuningExperiment:
    def __init__(
        self,
        build_env,
        build_test_env,
        algo_cls,
        agent_cls,
        config: ExperimentConfig,
        seed=None,
    ) -> None:
        self.study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.trial.Trial):
            if seed:
                np.random.seed(seed)
                torch.random.manual_seed(seed)

            algo_config = config.algo if config.algo else {}
            agent_config = config.agent if config.agent else {}

            # set tuned parameters
            for param, value in config.algo_tune_params.items():
                if value.type == "float":
                    algo_config[param] = trial.suggest_float(
                        param, value.low, value.high
                    )
                elif value.type == "int":
                    algo_config[param] = trial.suggest_int(param, value.low, value.high)
                else:
                    raise NotImplementedError("Unknown tuning parameter type")

            for param, value in config.agent_tune_params.items():
                if value.type == "float":
                    agent_config[param] = trial.suggest_float(
                        param, value.low, value.high
                    )
                elif value.type == "int":
                    agent_config[param] = trial.suggest_int(
                        param, value.low, value.high
                    )
                else:
                    raise NotImplementedError("Unknown tuning parameter type")

            algo_config = DictConfig(config.algo)
            # agent_config = DictConfig(config.agent)

            env = build_env()
            test_env = build_test_env()

            obs_dim, action_dim = (
                env.single_observation_space.shape[-1],
                env.single_action_space.n,
            )

            env.reset(seed=seed)
            agent = agent_cls(obs_dim, action_dim, **agent_config)
            algo = algo_cls(agent, algo_config)
            end_metrics = algo.train(env, test_env, save=False, verbose=False)
            return end_metrics["avg_episodic_returns"]

        self.objective = objective

    def run(self, n_trials=None, timeout=None, n_jobs=1):
        self.study.optimize(
            self.objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs
        )
        return self.study
