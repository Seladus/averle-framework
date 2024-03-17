import numpy as np
import scipy.signal as signal

from collections import namedtuple, deque
from typing import Sequence


Batch = namedtuple("Transition", ["obs", "next_obs", "action", "reward", "done"])


class ReplayBuffer:
    def __init__(self, observation_shape, max_size=int(2e6), n_envs=1, seed=42):
        self.rng = np.random.default_rng(seed)

        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.observation_shape = observation_shape

        self.n_envs = n_envs

        self.observation = np.zeros((n_envs, self.max_size) + observation_shape)
        self.next_observation = np.zeros((n_envs, self.max_size) + observation_shape)
        self.action = np.zeros(
            (n_envs, self.max_size),
            dtype="int",
        )
        self.reward = np.zeros((n_envs, self.max_size))
        self.done = np.zeros((n_envs, self.max_size))
        self.episode_ended = np.zeros((n_envs, self.max_size))
        self.path_start_idx = np.zeros((n_envs), dtype=np.int32)

    def add(self, state, action, next_state, reward, done, truncated):
        self.observation[:, self.ptr] = state
        self.action[:, self.ptr] = action
        self.next_observation[:, self.ptr] = next_state
        self.reward[:, self.ptr] = reward
        self.done[:, self.ptr] = done

        self.size = min(self.size + 1, self.max_size)

        path_ended = np.logical_or(done, truncated)
        self.episode_ended[:, self.ptr] = path_ended

        self.ptr = (self.ptr + 1) % self.max_size
        self.path_start_idx[path_ended] = self.ptr

    def _sample_idxs(self, batch_size):
        ind = self.rng.integers(0, self.size * self.n_envs, size=batch_size)
        return ind // self.size, ind % self.size

    def _sample_sequences(self, env_inds, inds, sequence_length):
        nb_seq = inds.shape[0]

        start = inds
        end = start + sequence_length - 1

        obs = np.zeros((nb_seq, sequence_length) + self.observation_shape)
        next_obs = np.zeros((nb_seq, sequence_length) + self.observation_shape)
        actions = np.zeros((nb_seq, sequence_length), dtype=np.int32)
        rewards = np.zeros((nb_seq, sequence_length))
        dones = np.zeros((nb_seq, sequence_length))

        for i in range(nb_seq):
            env_id = env_inds[i]
            s = start[i]
            e = end[i]

            done_in_seq = self.episode_ended[env_id, s:e].nonzero()[0]
            if done_in_seq.any():
                e = s + done_in_seq[0]

                done_before = self.episode_ended[
                    env_id, np.maximum(e - sequence_length, 0) : e
                ][::-1].nonzero()[0]
                if done_before.any():
                    s = e - done_before[0]
                s = np.maximum(e - sequence_length, 0)

                obs[i, 0 : e - s] = self.observation[env_id, s:e]
                next_obs[i, 0 : e - s] = self.next_observation[env_id, s:e]
                actions[i, 0 : e - s] = self.action[env_id, s:e]
                rewards[i, 0 : e - s] = self.reward[env_id, s:e]
                dones[i, 0 : e - s] = self.done[env_id, s:e]
        return Batch(
            obs,
            next_obs,
            np.expand_dims(actions, -1),
            np.expand_dims(rewards, -1),
            np.expand_dims(dones, -1),
        )

    def sample_sequences(self, batch_size, sequence_length):
        env_inds, inds = self._sample_idxs(batch_size)
        return self._sample_sequences(env_inds, inds, sequence_length)

    def sample_sequences_batch(self, nb_batch, batch_size, sequence_length):
        batch = self.sample_sequences(batch_size * nb_batch, sequence_length)
        return Batch(
            batch.obs.reshape(
                (
                    nb_batch,
                    batch_size,
                    sequence_length,
                )
                + self.observation_shape
            ),
            batch.next_obs.reshape(
                (
                    nb_batch,
                    batch_size,
                    sequence_length,
                )
                + self.observation_shape
            ),
            batch.action.reshape((nb_batch, batch_size, sequence_length, 1)),
            batch.reward.reshape((nb_batch, batch_size, sequence_length, 1)),
            batch.done.reshape((nb_batch, batch_size, sequence_length, 1)),
        )

    def sample(self, batch_size):
        env_idx, ind = self._sample_idxs(batch_size)
        return Batch(
            self.observation[env_idx, ind],
            self.next_observation[env_idx, ind],
            self.action[env_idx, ind],
            self.reward[env_idx, ind],
            self.done[env_idx, ind],
        )

    def sample_batch(self, nb_batch, batch_size):
        batch = self.sample(batch_size * nb_batch)
        return Batch(
            batch.obs.reshape(
                (
                    nb_batch,
                    batch_size,
                )
                + self.observation_shape
            ),
            batch.next_obs.reshape(
                (
                    nb_batch,
                    batch_size,
                )
                + self.observation_shape
            ),
            batch.action.reshape((nb_batch, batch_size, 1)),
            batch.reward.reshape((nb_batch, batch_size, 1)),
            batch.done.reshape((nb_batch, batch_size, 1)),
        )
