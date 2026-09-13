"""D3IL Avoiding as a chunked, normalized gym env.

Wraps the DPPO fork's ``avoiding-m5`` (obs = [desired_xy, measured_xy], action = delta of the
desired EE position in [-0.01, 0.01]^2) so that a policy sees observations in [-1, 1]^4 and emits an
action chunk of shape (act_steps, 2) in [-1, 1], executed open-loop inside one ``step()``.

Notes
-----
* The fork returns ``obs[:2]`` one step stale (``prev_action`` is updated *after*
  ``get_observation``). We rebuild the observation from ``env.prev_action`` so the policy sees the
  same (des_t, c_t) pairing it was trained on.
* Episodes are fixed length (``ep_len`` env steps). The fork never terminates early; success is
  "reached the finish line at some point and never touched a pillar" (D3IL definition).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import gym
import gym_avoiding  # noqa: F401  (registers avoiding-v0 / avoiding-m5..m8)
import numpy as np
from gym import spaces
from gym.envs import make as gym_make

ENV_ID = "avoiding-m5"  # m5..m8 differ only in the extra +1 for passing a specific hole
EP_LEN = 100  # env steps per episode (DPPO setting); registry TimeLimit is 150
ACTION_BOUND = 0.01  # |delta desired EE pos| per env step, from the fork's action_space
MID_X = 0.5  # maze centre line
L1_Y = -0.1  # y of the first pillar row
LEVEL_DY = 0.18  # y spacing between pillar rows
GOAL_Y = L1_Y + 2.5 * LEVEL_DY  # env.goal_ypos
DEFAULT_OFFSET = 0.075  # x spacing of the pillars, fork's `obstacle_offset`
DEFAULT_L3_RADIUS = 0.025  # radius of the last-row cylinders
DATA_DIR = Path(os.environ.get("DPPO_DATA_DIR", "data"))


def maze_params() -> tuple[float, float]:
    """Pillar spacing / last-row radius, overridable per run to make the maze harder.

    AVOID_OFFSET=0.06 AVOID_L3_RADIUS=0.035 just eval m1 --device cuda
    """
    return (
        float(os.environ.get("AVOID_OFFSET", DEFAULT_OFFSET)),
        float(os.environ.get("AVOID_L3_RADIUS", DEFAULT_L3_RADIUS)),
    )


def _pillar_xy(offset: float) -> list[list[float]]:
    """(6, 2) pillar centres for a given spacing, in fork order (l1, l2 x2, l3 x3)."""
    return [
        [MID_X, L1_Y],
        [MID_X - offset, L1_Y + LEVEL_DY],
        [MID_X + offset, L1_Y + LEVEL_DY],
        [MID_X - 2 * offset, L1_Y + 2 * LEVEL_DY],
        [MID_X, L1_Y + 2 * LEVEL_DY],
        [MID_X + 2 * offset, L1_Y + 2 * LEVEL_DY],
    ]


def _apply_maze_params(offset: float, l3_radius: float) -> None:
    """Reshape the pillars the D3IL env builds its scene from.

    ``avoiding.py`` runs ``obj_list = get_obj_list()`` once at import, so the geometry has to be
    replaced on that module attribute; patching ``get_obj_list`` alone would come too late.
    """
    from gym_avoiding.envs import avoiding
    from gym_avoiding.envs.objects.avoiding_objects import get_obj_list

    objs = get_obj_list()
    names = ["l1_obs", "l2_top_obs", "l2_bottom_obs", "l3_top_obs", "l3_mid_obs", "l3_bottom_obs"]
    xy = dict(zip(names, _pillar_xy(offset)))
    for o in objs:
        if o.name in xy:
            o.init_pos = [*xy[o.name], 0]
        if o.name.startswith("l3_"):
            o.size = [l3_radius, o.size[1]]
    avoiding.obj_list = objs


@dataclass
class Normalizer:
    """min/max -> [-1, 1] (same convention as DPPO's normalization.npz)."""

    obs_min: np.ndarray
    obs_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        d = np.load(path)
        return cls(
            *(d[k].astype(np.float32) for k in ("obs_min", "obs_max", "action_min", "action_max"))
        )

    @classmethod
    def default(cls) -> Normalizer:
        """Workspace bounds fallback when normalization.npz is not available."""
        lo = np.array([0.2, -0.45, 0.2, -0.45], np.float32)
        hi = np.array([0.8, 0.45, 0.8, 0.45], np.float32)
        a = np.full(2, ACTION_BOUND, np.float32)
        return cls(lo, hi, -a, a)

    def obs(self, x: np.ndarray) -> np.ndarray:
        return (2 * ((x - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)).astype(
            np.float32
        )

    def act(self, a: np.ndarray) -> np.ndarray:  # raw -> [-1, 1]
        return (
            2 * ((a - self.action_min) / (self.action_max - self.action_min + 1e-6) - 0.5)
        ).astype(np.float32)

    def unact(self, a: np.ndarray) -> np.ndarray:  # [-1, 1] -> raw
        return ((a + 1) / 2 * (self.action_max - self.action_min) + self.action_min).astype(
            np.float32
        )

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("obs_min", "obs_max", "action_min", "action_max")}


def load_normalizer(mode_dir: str | Path | None) -> Normalizer:
    p = Path(mode_dir) / "normalization.npz" if mode_dir else None
    return Normalizer.load(p) if p and p.exists() else Normalizer.default()


@dataclass
class EnvConfig:
    env_id: str = ENV_ID
    act_steps: int = 4  # actions executed per policy call (T_a)
    ep_len: int = EP_LEN
    render: bool = False
    norm_dir: str | None = str(DATA_DIR / "d3il" / "avoid_m1")
    extra: dict = field(default_factory=dict)


class AvoidingChunkEnv(gym.Env):
    metadata: ClassVar[dict] = {"render.modes": ["rgb_array"]}

    def __init__(self, cfg: EnvConfig | None = None, norm: Normalizer | None = None):
        self.cfg = cfg or EnvConfig()
        self.norm = norm or load_normalizer(self.cfg.norm_dir)
        offset, l3_radius = maze_params()
        if (offset, l3_radius) != (DEFAULT_OFFSET, DEFAULT_L3_RADIUS):
            _apply_maze_params(offset, l3_radius)
        # max_steps_per_episode > ep_len so the inner env never flags done before we do
        self.env = gym_make(
            self.cfg.env_id, render=self.cfg.render, max_steps_per_episode=self.cfg.ep_len + 5
        )
        self.u = self.env.unwrapped
        # the fork hardcodes the gap boundaries from the default spacing; keep mode_encoding honest
        self.u.l2_top_xpos = MID_X - offset
        self.u.l2_bottom_xpos = MID_X + offset
        self.u.l3_top_xpos = MID_X - 2 * offset
        self.u.l3_bottom_xpos = MID_X + 2 * offset
        self.u.obj_xy_list = _pillar_xy(offset)
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.cfg.act_steps, 2), np.float32)
        self.t = 0
        self.reached_goal = False
        self.ee_traj: list[np.ndarray] = []

    # -- helpers -------------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        raw = np.hstack([self.u.prev_action, self.u.robot.current_c_pos[:2]]).astype(np.float32)
        return self.norm.obs(raw)

    def _info(self) -> dict:
        collision = bool(self.u.collision)
        return {
            "t": self.t,
            "reached_goal": self.reached_goal,
            "collision": collision,
            "success": self.reached_goal and not collision,
            "mode_encoding": self.u.mode_encoding.astype(np.int8).copy(),
            "ee_pos": self.u.robot.current_c_pos[:2].astype(np.float32).copy(),
        }

    # -- gym API -------------------------------------------------------------------------------
    def seed(self, seed: int | None = None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed: int | None = None, options: dict | None = None, **_):
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        self.t, self.reached_goal = 0, False
        self.ee_traj = [self.u.robot.current_c_pos[:2].astype(np.float32).copy()]
        return self._obs()

    def step(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, np.float32).reshape(-1, 2)
        raw_chunk = np.clip(self.norm.unact(chunk), -ACTION_BOUND, ACTION_BOUND)
        total_r = 0.0
        for a in raw_chunk:
            _, r, _, _ = self.env.step(a)
            total_r += float(r)
            self.t += 1
            pos = self.u.robot.current_c_pos[:2]
            self.ee_traj.append(pos.astype(np.float32).copy())
            self.reached_goal |= bool(pos[1] > GOAL_Y)
            if self.t >= self.cfg.ep_len:
                break
        done = self.t >= self.cfg.ep_len
        info = self._info()
        if done:
            info["ee_traj"] = np.stack(self.ee_traj)
        return self._obs(), total_r, done, info

    def render(self, mode="rgb_array", **kw):
        return self.env.render(mode=mode, **kw)

    def close(self):
        self.env.close()


def obstacle_xy() -> np.ndarray:
    """(6, 2) pillar centres, for plotting."""
    return np.asarray(_pillar_xy(maze_params()[0]), np.float32)


def make_env_fn(cfg: EnvConfig, norm: Normalizer, seed: int):
    def _fn():
        env = AvoidingChunkEnv(cfg, norm)
        env.seed(seed)
        return env

    return _fn


def make_vec_env(cfg: EnvConfig, norm: Normalizer, n_envs: int, seed: int = 0):
    """AsyncVectorEnv with spawn (MuJoCo + fork = trouble). Auto-resets on done (gym 0.22)."""
    from gym.vector import AsyncVectorEnv, SyncVectorEnv

    fns = [make_env_fn(cfg, norm, seed + i) for i in range(n_envs)]
    if n_envs == 1:
        return SyncVectorEnv(fns)
    return AsyncVectorEnv(fns, shared_memory=False, context="spawn")


def smoke_test():
    """`d3il-flow-smoke`: build one env, drive straight into the first pillar, print timings."""
    import time

    env = AvoidingChunkEnv(EnvConfig(norm_dir=None))
    obs = env.reset()
    print("obs", obs, "act_space", env.action_space.shape)
    t0 = time.time()
    chunk = np.tile([0.0, 0.6], (env.cfg.act_steps, 1))  # +y at 60% of max speed
    done, R = False, 0.0
    while not done:
        obs, r, done, info = env.step(chunk)
        R += r
    dt = time.time() - t0
    print(f"episode: {env.cfg.ep_len} env steps in {dt:.1f}s ({env.cfg.ep_len / dt:.0f} steps/s)")
    print({k: v for k, v in info.items() if k != "ee_traj"}, "return", round(R, 2))
    env.close()


if __name__ == "__main__":
    smoke_test()
