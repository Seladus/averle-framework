import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
