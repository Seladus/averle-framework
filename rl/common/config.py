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
