import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RecurrentAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def critic(self, state, hidden, terminal=None):
        raise NotImplementedError()

    def actor(self, state, hidden, terminal=None):
        raise NotImplementedError()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h, recurrent_layers):
        super().__init__()
        self.recurrent_layers = recurrent_layers
        self.h = h
        self.lstm = nn.LSTM(state_dim, h, num_layers=recurrent_layers)
        self.layer_hidden = nn.Linear(h, h)
        self.layer_policy_logits = nn.Linear(h, action_dim)
        self.action_dim = action_dim
        self.hidden_cell = None

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (
            torch.zeros(self.recurrent_layers, batch_size, self.h).to(device),
            torch.zeros(self.recurrent_layers, batch_size, self.h).to(device),
        )
        return self.hidden_cell

    def forward(self, state, hidden, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if batch_size != hidden[0].shape[1]:
            hidden = self.get_init_state(batch_size, device)
        if terminal is not None:
            hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1) for value in hidden
            ]
        x, new_hidden = self.lstm(state, hidden)
        hidden_out = F.elu(self.layer_hidden(x))
        policy_logits_out = self.layer_policy_logits(hidden_out)
        probs = F.softmax(policy_logits_out, dim=-1)
        return probs, new_hidden


class Critic(nn.Module):
    def __init__(self, state_dim, h, recurrent_layers):
        super().__init__()
        self.recurrent_layers = recurrent_layers
        self.h = h
        self.layer_lstm = nn.LSTM(state_dim, h, num_layers=recurrent_layers)
        self.layer_hidden = nn.Linear(h, h)
        self.layer_value = nn.Linear(h, 1)
        self.hidden_cell = None

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (
            torch.zeros(self.recurrent_layers, batch_size, self.h).to(device),
            torch.zeros(self.recurrent_layers, batch_size, self.h).to(device),
        )
        return self.hidden_cell

    def forward(self, state, hidden, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if batch_size != hidden[0].shape[1]:
            hidden = self.get_init_state(batch_size, device)
        if terminal is not None:
            hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1) for value in hidden
            ]
        x, new_hidden = self.layer_lstm(state, hidden)
        hidden_out = F.elu(self.layer_hidden(x))
        value_out = self.layer_value(hidden_out)
        return value_out, new_hidden


class SimpleRecurrentAgent(RecurrentAgent):
    def __init__(self, state_dim, action_dim, h=64, recurrent_layers=1) -> None:
        super().__init__()
        self.actor_net = Actor(state_dim, action_dim, h, recurrent_layers)
        self.critic_net = Critic(state_dim, h, recurrent_layers)

    def get_init_state(self, batch_size, device):
        return (
            self.actor_net.get_init_state(batch_size, device),
            self.critic_net.get_init_state(batch_size, device),
        )

    def critic(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        value_out, critic_hidden = self.critic_net(state, critic_hidden, terminal)
        return value_out, (actor_hidden, critic_hidden)

    def actor(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        action_dist, actor_hidden = self.actor_net(state, actor_hidden, terminal)
        return action_dist, (actor_hidden, critic_hidden)


class SharedRecurrentAgent(RecurrentAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        f_enc=[],
        f_hidden=[64],
        f_lstm=64,
        n_recurrent=1,
        f_actor=[],
        f_critic=[],
    ):
        super().__init__()

        self.n_recurrent = n_recurrent
        self.f_lstm = f_lstm
        f_enc = [state_dim] + f_enc
        f_hidden = [f_lstm] + f_hidden
        f_actor = [f_hidden[-1]] + f_actor + [action_dim]
        f_critic = [f_hidden[-1]] + f_critic + [1]
        activation = nn.ELU

        # encoding layers
        enc = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_enc) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_enc[:-1], f_enc[1:]))
        ]
        self.enc = nn.Sequential(*enc)
        # recurrent layers
        self.recurrent = nn.LSTM(f_enc[-1], f_lstm, num_layers=n_recurrent)
        # hidden
        h = [
            nn.Sequential(nn.Linear(i, o), activation())
            for idx, (i, o) in enumerate(zip(f_hidden[:-1], f_hidden[1:]))
        ]
        self.hidden_dense = nn.Sequential(*h)
        # output heads
        actor_head = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_actor) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_actor[:-1], f_actor[1:]))
        ]
        self.actor_head = nn.Sequential(*actor_head)
        critic_head = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_critic) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_critic[:-1], f_critic[1:]))
        ]
        self.critic_head = nn.Sequential(*critic_head)

    def get_init_state(self, batch_size, device):
        hidden = (
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
        )
        return (hidden, hidden)

    def actor(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != actor_hidden[0].shape[1]:
            actor_hidden = self.get_init_state(batch_size, device)[0]
        if terminal is not None:
            actor_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in actor_hidden
            ]
        x = self.enc(state)
        x, new_actor_hidden = self.recurrent(x, actor_hidden)
        x = self.hidden_dense(x)
        logits = self.actor_head(x)
        probs = F.softmax(logits, dim=-1)
        return probs, (new_actor_hidden, critic_hidden)

    def critic(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != critic_hidden[0].shape[1]:
            critic_hidden = self.get_init_state(batch_size, device)[1]
        if terminal is not None:
            critic_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in critic_hidden
            ]
        x = self.enc(state)
        x, new_critic_hidden = self.recurrent(x, critic_hidden)
        x = self.hidden_dense(x)
        value = self.critic_head(x)
        return value, (actor_hidden, new_critic_hidden)


class SplittedRecurrentAgent(RecurrentAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        f_actor_enc=[],
        f_critic_enc=[],
        f_lstm=64,
        n_recurrent=1,
        f_actor_head=[64],
        f_critic_head=[64],
    ):
        super().__init__()

        self.n_recurrent = n_recurrent
        self.f_lstm = f_lstm
        f_actor_enc = [state_dim] + f_actor_enc
        f_critic_enc = [state_dim] + f_critic_enc
        f_actor_head = [f_lstm] + f_actor_head + [action_dim]
        f_critic_head = [f_lstm] + f_critic_head + [1]
        activation = nn.ELU

        # encoding layers
        actor_enc = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_actor_enc) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_actor_enc[:-1], f_actor_enc[1:]))
        ]
        critic_enc = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_critic_enc) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_critic_enc[:-1], f_critic_enc[1:]))
        ]
        self.actor_enc = nn.Sequential(*actor_enc)
        self.critic_enc = nn.Sequential(*critic_enc)
        # recurrent layers
        self.actor_recurrent = nn.LSTM(f_actor_enc[-1], f_lstm, num_layers=n_recurrent)
        self.critic_recurrent = nn.LSTM(
            f_critic_enc[-1], f_lstm, num_layers=n_recurrent
        )
        # output heads
        actor_head = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_actor_head) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_actor_head[:-1], f_actor_head[1:]))
        ]
        self.actor_head = nn.Sequential(*actor_head)
        critic_head = [
            nn.Sequential(nn.Linear(i, o), activation())
            if idx < (len(f_critic_head) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_critic_head[:-1], f_critic_head[1:]))
        ]
        self.critic_head = nn.Sequential(*critic_head)

    def get_init_state(self, batch_size, device):
        actor_hidden = (
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
        )
        critic_hidden = (
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
        )
        return (actor_hidden, critic_hidden)

    def actor(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != actor_hidden[0].shape[1]:
            actor_hidden = self.get_init_state(batch_size, device)[0]
        if terminal is not None:
            actor_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in actor_hidden
            ]
        x = self.actor_enc(state)
        # x = state
        x, new_actor_hidden = self.actor_recurrent(x, actor_hidden)
        logits = self.actor_head(x)
        probs = F.softmax(logits, dim=-1)
        return probs, (new_actor_hidden, critic_hidden)

    def critic(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != critic_hidden[0].shape[1]:
            critic_hidden = self.get_init_state(batch_size, device)[1]
        if terminal is not None:
            critic_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in critic_hidden
            ]
        x = self.critic_enc(state)
        x, new_critic_hidden = self.critic_recurrent(x, critic_hidden)
        value = self.critic_head(x)
        return value, (actor_hidden, new_critic_hidden)


class DropoutSplittedRecurrentAgent(RecurrentAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        f_actor_enc=[],
        f_critic_enc=[],
        f_lstm=64,
        n_recurrent=1,
        f_actor_head=[64],
        f_critic_head=[64],
        p=0.5,
    ):
        super().__init__()

        self.n_recurrent = n_recurrent
        self.f_lstm = f_lstm
        f_actor_enc = [state_dim] + f_actor_enc
        f_critic_enc = [state_dim] + f_critic_enc
        f_actor_head = [f_lstm] + f_actor_head + [action_dim]
        f_critic_head = [f_lstm] + f_critic_head + [1]
        activation = nn.ELU

        # encoding layers
        actor_enc = [
            nn.Sequential(nn.Linear(i, o), nn.Dropout(p=p), activation())
            if idx < (len(f_actor_enc) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_actor_enc[:-1], f_actor_enc[1:]))
        ]
        critic_enc = [
            nn.Sequential(nn.Linear(i, o), nn.Dropout(p=p), activation())
            if idx < (len(f_critic_enc) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_critic_enc[:-1], f_critic_enc[1:]))
        ]
        self.actor_enc = nn.Sequential(*actor_enc)
        self.critic_enc = nn.Sequential(*critic_enc)
        # recurrent layers
        self.actor_recurrent = nn.LSTM(f_actor_enc[-1], f_lstm, num_layers=n_recurrent)
        self.critic_recurrent = nn.LSTM(
            f_critic_enc[-1], f_lstm, num_layers=n_recurrent
        )
        # output heads
        actor_head = [
            nn.Sequential(nn.Linear(i, o), nn.Dropout(p=p), activation())
            if idx < (len(f_actor_head) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_actor_head[:-1], f_actor_head[1:]))
        ]
        self.actor_head = nn.Sequential(*actor_head)
        critic_head = [
            nn.Sequential(nn.Linear(i, o), nn.Dropout(p=p), activation())
            if idx < (len(f_critic_head) - 2)
            else nn.Linear(i, o)
            for idx, (i, o) in enumerate(zip(f_critic_head[:-1], f_critic_head[1:]))
        ]
        self.critic_head = nn.Sequential(*critic_head)

    def get_init_state(self, batch_size, device):
        actor_hidden = (
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
        )
        critic_hidden = (
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
            torch.zeros(self.n_recurrent, batch_size, self.f_lstm).to(device),
        )
        return (actor_hidden, critic_hidden)

    def actor(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != actor_hidden[0].shape[1]:
            actor_hidden = self.get_init_state(batch_size, device)[0]
        if terminal is not None:
            actor_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in actor_hidden
            ]
        x = self.actor_enc(state)
        # x = state
        x, new_actor_hidden = self.actor_recurrent(x, actor_hidden)
        logits = self.actor_head(x)
        probs = F.softmax(logits, dim=-1)
        return probs, (new_actor_hidden, critic_hidden)

    def critic(self, state, hidden, terminal=None):
        actor_hidden, critic_hidden = hidden
        batch_size = state.shape[1]
        device = state.device
        if batch_size != critic_hidden[0].shape[1]:
            critic_hidden = self.get_init_state(batch_size, device)[1]
        if terminal is not None:
            critic_hidden = [
                value * (1.0 - terminal).reshape(1, batch_size, 1)
                for value in critic_hidden
            ]
        x = self.critic_enc(state)
        x, new_critic_hidden = self.critic_recurrent(x, critic_hidden)
        value = self.critic_head(x)
        return value, (actor_hidden, new_critic_hidden)
