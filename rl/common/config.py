import yaml


class Config:
    def __init__(self, path="./config.yml") -> None:
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)
        for key, value in self.config.items():
            self.__setattr__(key, value)

    def dict(self):
        return self.config

    def __getattr__(self, __name: str):
        return None


class DictConfig(Config):
    def __init__(self, dict_config={}) -> None:
        self.config = dict_config
        for key, value in self.config.items():
            self.__setattr__(key, value)


class ExperimentConfig(Config):
    def __init__(self, path="./config.yml") -> None:
        super().__init__(path)

        self.algo_tune_params = {}
        for key, value in self.config["algo"].items():
            if type(value) is dict and "type" in value.keys():  # param to be tuned
                self.algo_tune_params[key] = DictConfig(value)

        self.agent_tune_params = {}
        for key, value in self.config["agent"].items():
            if type(value) is dict and "type" in value.keys():  # param to be tuned
                self.agent_tune_params[key] = DictConfig(value)
