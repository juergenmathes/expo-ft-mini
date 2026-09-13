"""(obs_t, a_{t:t+T_a}) windows from a DPPO-style stitched npz.

npz layout (what `just data` downloads for avoid_m1/m2/m3):
    states        (N, 4)   float   [des_x, des_y, ee_x, ee_y], normalized to [-1, 1]
    actions       (N, 2)   float   delta desired EE xy, normalized to [-1, 1]
    traj_lengths  (E,)     int     episode lengths; sum == N

If arrays are not normalized (|x| > 1), a Normalizer is applied on load so raw exports work too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .env import Normalizer


class ChunkDataset(Dataset):
    def __init__(self, path: str | Path, act_steps: int = 4, norm: Normalizer | None = None):
        d = np.load(path, allow_pickle=False)
        states = d["states"].astype(np.float32)
        actions = d["actions"].astype(np.float32)
        lens = d["traj_lengths"].astype(np.int64)
        assert lens.sum() == len(states) == len(actions), (
            "traj_lengths must partition states/actions"
        )

        if norm is not None and (np.abs(states).max() > 1.05 or np.abs(actions).max() > 1.05):
            states, actions = norm.obs(states), norm.act(actions)

        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        idx = [
            s + t for s, L in zip(starts, lens) for t in range(L - act_steps + 1)
        ]  # DPPO: no padding
        self.index = np.asarray(idx, np.int64)
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.act_steps = act_steps
        self.n_episodes = len(lens)
        self.obs_dim, self.act_dim = states.shape[1], actions.shape[1]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        s = int(self.index[i])
        return self.states[s], self.actions[s : s + self.act_steps]  # (obs_dim,), (T_a, act_dim)

    def summary(self) -> str:
        return (
            f"{self.n_episodes} episodes, {len(self)} windows, obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim}, T_a={self.act_steps}, "
            f"state range [{self.states.min():.2f}, {self.states.max():.2f}], "
            f"action range [{self.actions.min():.2f}, {self.actions.max():.2f}]"
        )
