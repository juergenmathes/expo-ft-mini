"""Roll out a FlowPolicy in parallel envs; report success / collision / D3IL mode entropy; plot.

d3il-flow-eval --ckpt outputs/avoid_m1_flow/ckpt.pt --n-episodes 200 --n-envs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .env import GOAL_Y, EnvConfig, make_vec_env, maze_params, obstacle_xy
from .model import FlowPolicy


def mode_entropy(modes: np.ndarray) -> tuple[int, float]:
    """D3IL metric: entropy (log base 24) of the route distribution over the 9-bit mode encodings."""
    if len(modes) == 0:
        return 0, 0.0
    codes = modes.astype(np.int64).dot(1 << np.arange(modes.shape[-1]))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return len(counts), float(-(p * np.log(p) / np.log(24)).sum())


def rollout(policy: FlowPolicy, cfg: EnvConfig, n_episodes: int, n_envs: int, seed: int = 0):
    venv = make_vec_env(cfg, policy.norm, n_envs, seed)
    obs = venv.reset()
    episodes: list[dict] = []
    t0 = time.time()
    while len(episodes) < n_episodes:
        chunks = policy.act(obs)  # (n_envs, T_a, 2)
        obs, _, done, infos = venv.step(
            chunks
        )  # gym 0.22 autoresets; info carries the terminal stats
        for d, info in zip(done, infos):
            if d and len(episodes) < n_episodes:
                episodes.append(
                    {
                        k: info[k]
                        for k in (
                            "success",
                            "collision",
                            "reached_goal",
                            "mode_encoding",
                            "ee_traj",
                        )
                    }
                )
    venv.close()
    return episodes, time.time() - t0


def summarize(episodes: list[dict]) -> dict:
    offset, l3_radius = maze_params()
    succ = np.array([e["success"] for e in episodes])
    modes_ok = (
        np.stack([e["mode_encoding"] for e in episodes if e["success"]])
        if succ.any()
        else np.zeros((0, 9))
    )
    n_modes, ent = mode_entropy(modes_ok)
    return {
        "episodes": len(episodes),
        "success_rate": float(succ.mean()),
        "collision_rate": float(np.mean([e["collision"] for e in episodes])),
        "reached_goal_rate": float(np.mean([e["reached_goal"] for e in episodes])),
        "distinct_modes_success": n_modes,
        "mode_entropy_success": ent,  # 1.0 == uniform over all 24 routes
        "maze_offset": offset,
        "maze_l3_radius": l3_radius,
    }


def plot(episodes: list[dict], path: Path, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for e in episodes:
        tr = e["ee_traj"]
        ax.plot(
            tr[:, 1], tr[:, 0], lw=0.8, alpha=0.6, color="tab:green" if e["success"] else "tab:red"
        )
    l3_r = maze_params()[1]
    for (x, y), r in zip(obstacle_xy(), [0.03, 0.025, 0.025, l3_r, l3_r, l3_r]):
        ax.add_patch(plt.Circle((y, x), r, color="k"))
    ax.axvline(GOAL_Y, color="tab:green", lw=2, ls="--", label="finish line")
    ax.set_xlim(-0.35, 0.45), ax.set_ylim(0.8, 0.2)  # top-down view, robot base at the bottom
    (
        ax.set_xlabel("y (towards goal)"),
        ax.set_ylabel("x"),
        ax.set_aspect("equal"),
        ax.legend(loc="upper left"),
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] {path}")


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", default="outputs/avoid_m1_flow/ckpt.pt")
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--flow-steps", type=int, default=8, help="ODE integration steps")
    p.add_argument("--method", default="euler", choices=["euler", "midpoint"])
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--ep-len", type=int, default=100)
    p.add_argument("--env-id", default="avoiding-m5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu is fine: the head is tiny, sim dominates")
    p.add_argument(
        "--out", default=None, help="dir for metrics.json + rollouts.png (default: ckpt dir)"
    )
    return p.parse_args()


def main():
    a = parse()
    policy = FlowPolicy.load(
        a.ckpt, device=a.device, use_ema=not a.no_ema, n_steps=a.flow_steps, method=a.method
    )
    cfg = EnvConfig(env_id=a.env_id, act_steps=policy.model.cfg.act_steps, ep_len=a.ep_len)
    episodes, dt = rollout(policy, cfg, a.n_episodes, a.n_envs, a.seed)
    m = summarize(episodes)
    m["wall_s"] = round(dt, 1)
    print(json.dumps(m, indent=2))
    out = Path(a.out or Path(a.ckpt).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    plot(
        episodes,
        out / "rollouts.png",
        f"{Path(a.ckpt).parent.name}: success {m['success_rate']:.2f}, H24 {m['mode_entropy_success']:.2f}",
    )


if __name__ == "__main__":
    main()
