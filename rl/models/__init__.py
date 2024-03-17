from torch import nn


class QAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def q(self, state):
        raise NotImplementedError()


class ActorCriticAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def critic(self, state):
        raise NotImplementedError()

    def actor(self, state):
        raise NotImplementedError()


class RecurrentActorCriticAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def get_init_state(self, batch_size, device):
        raise NotImplementedError()

    def critic(self, state, hidden, terminal=None):
        raise NotImplementedError()

    def actor(self, state, hidden, terminal=None):
        raise NotImplementedError()
