import torch
from torch.nn.utils.rnn import pad_sequence


class RecurrentRolloutBuffer:
    def __init__(self, device) -> None:
        self.device = device
        self.trajectory_data = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "values": [],
            "terminals": [],
            "actor_hidden_states": [],
            "actor_cell_states": [],
            "critic_hidden_states": [],
            "critic_cell_states": [],
        }

    def add(self, hidden, state, value, action, logprob, reward, terminal):
        actor_hidden, critic_hidden = hidden
        self.trajectory_data["actor_hidden_states"].append(
            actor_hidden[0].permute(1, 0, 2)
        )
        self.trajectory_data["actor_cell_states"].append(
            actor_hidden[1].permute(1, 0, 2)
        )
        self.trajectory_data["critic_hidden_states"].append(
            critic_hidden[0].permute(1, 0, 2)
        )
        self.trajectory_data["critic_cell_states"].append(
            critic_hidden[1].permute(1, 0, 2)
        )

        self.trajectory_data["states"].append(state)
        self.trajectory_data["values"].append(value)
        self.trajectory_data["actions"].append(action)
        self.trajectory_data["logprobs"].append(logprob)
        self.trajectory_data["rewards"].append(reward)
        self.trajectory_data["terminals"].append(terminal)

    def finish_rollout(self, final_value, done, gamma, gae_lambda):
        trajectory_tensors = {
            key: torch.stack(value).to(self.device)
            for key, value in self.trajectory_data.items()
        }

        epoch_steps = trajectory_tensors["states"].shape[0]
        values = trajectory_tensors["values"]
        rewards = trajectory_tensors["rewards"]
        dones = trajectory_tensors["terminals"]
        done = torch.Tensor(done).to(self.device)

        # compute advantages and returns
        with torch.no_grad():
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(epoch_steps)):
                if t == epoch_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = final_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        trajectory_tensors["advantages"] = advantages
        trajectory_tensors["returns"] = returns
        self.trajectory_tensors = trajectory_tensors

    def build_sequences(self, seq_len):
        # split rollout in episodes
        n_envs = self.trajectory_tensors["states"].shape[1]
        trajectory_episodes = {key: [] for key in self.trajectory_tensors.keys()}

        terminals = self.trajectory_tensors["terminals"].clone()
        terminals[-1] = 1
        terminals[0] = 1

        len_episodes = []
        for i in range(n_envs):
            split_points = (terminals[:, i] == 1).nonzero() + 1
            split_lens = split_points[1:] - split_points[:-1]
            split_lens[0] += 1

            len_episode = [split_len.item() for split_len in split_lens]
            len_episodes += len_episode

            for key, value in self.trajectory_tensors.items():
                split = torch.split(value[:, i], len_episode)
                trajectory_episodes[key] += split

        # split episodes in sequences of seq_len
        nb_episodes = len(len_episodes)
        sequences_idxs = []
        for i in range(nb_episodes):
            l = len_episodes[i]
            ep_seq_idxs = []
            for start in range(0, l, seq_len):
                end = min(start + seq_len, l)
                ep_seq_idxs.append(torch.arange(start, end))
            sequences_idxs.append(ep_seq_idxs)

        trajectory_sequences = {key: [] for key in self.trajectory_tensors.keys()}

        for key, value in trajectory_sequences.items():
            for i, seq_idxs in enumerate(sequences_idxs):
                for idxs in seq_idxs:
                    trajectory_sequences[key].append(trajectory_episodes[key][i][idxs])
            trajectory_sequences[key] = pad_sequence(trajectory_sequences[key])

        nb_seq = trajectory_sequences["states"].shape[1]
        sequences_idxs = [item for row in sequences_idxs for item in row]

        # build masks
        masks = torch.zeros((nb_seq, seq_len))
        for i, seq_idxs in enumerate(sequences_idxs):
            l = len(seq_idxs)
            masks[i, :l] = torch.ones(l)

        return (
            trajectory_sequences["states"],
            trajectory_sequences["actions"],
            trajectory_sequences["logprobs"],
            trajectory_sequences["advantages"],
            trajectory_sequences["returns"],
            trajectory_sequences["actor_hidden_states"],
            trajectory_sequences["actor_cell_states"],
            trajectory_sequences["critic_hidden_states"],
            trajectory_sequences["critic_cell_states"],
            masks.to(self.device),
        ), nb_seq
