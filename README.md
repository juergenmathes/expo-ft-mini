# d3il-flow

Flow-matching action policies on the **D3IL Avoiding** task (top-down, 6 pillars, 24 routes), built on
the DPPO fork of D3IL (`avoiding-m5`: 4-D state, 2-D delta action, reward, never terminates early).

```
.devcontainer/   Dockerfile (base→deps→dev→runtime) · devcontainer.json · set_me_up.sh
src/d3il_flow/   env.py (chunked gym wrapper + vec env) · data.py · model.py (CFM head) · train.py · eval.py
scripts/         download_data.sh (DPPO's pre-processed avoid_m1/m2/m3 npz + normalization)
third_party/     D3IL fork, cloned at a pinned commit by set_me_up.sh (gitignored)
```

## Spawn

```bash
git clone <this repo> d3il-flow && cd d3il-flow
export TORCH_EXTRA=cu128        # cu124 for Ampere/Ada, cu128 for Blackwell (sm_120); omit on CPU-only hosts
code .                          # → "Reopen in Container"  (or: devcontainer up --workspace-folder .)
```

`postCreate` runs `set_me_up.sh`: clones the D3IL fork, `uv sync`, picks the MuJoCo GL backend, runs a smoke episode.
Then:

```bash
just data m1          # train.npz (96 demos, 4-D obs / 2-D act, normalized) + normalization.npz
just train m1         # CFM-BC, 20k steps, ~2 min on CPU → outputs/avoid_m1_flow/ckpt.pt
just eval m1 --n-episodes 200 --n-envs 8 --flow-steps 8   # success / collision / mode entropy + rollouts.png
```

Without VS Code: `just docker dev && docker run -it --rm -v $PWD:/workspace d3il-flow:dev bash`.

## Env I/O (`d3il_flow.env.AvoidingChunkEnv`)

| | shape | meaning |
|---|---|---|
| obs | `(4,)` in [-1, 1] | `[des_x, des_y, ee_x, ee_y]` — commanded EE setpoint + measured EE position |
| action | `(T_a, 2)` in [-1, 1] | chunk of **Δ setpoint** per env step, unnormalized to ±0.01 m, executed open-loop |
| reward | float | fork's: +1/step past the finish line, −0.1/step after any collision (permanent) or outside x∈[0.2, 0.8] |
| done | bool | after `ep_len` env steps (100) — fixed length |
| info | dict | `success` (reached finish line ∧ no collision), `collision`, `reached_goal`, `mode_encoding (9,)`, `ee_pos`, `ee_traj (T+1, 2)` on done |

Under the hood: 35 MuJoCo substeps @ 1 ms per env step, Franka + Cartesian tracking controller, z fixed at 0.12 m.
~25 env steps/s per core → `n_envs` ≈ physical cores; `--n-envs 16 --n-episodes 500` ≈ 2 min.

Metric conventions follow D3IL: **success rate** and **mode entropy** (log base 24 over the route codes of
successful episodes; 1.0 = uniform over all 24 routes).

## Model

`FlowMLP` — residual MLP `v_θ(x_τ, τ | obs)` over the flattened chunk, OT-CFM loss, Euler/midpoint sampler,
EMA weights. ~200k params; the sim, not the head, is the bottleneck.

RL hooks already in `model.sample`:
- `x0=` → noise-space control (DSRL-style: RL picks x₀, head frozen)
- `return_path=True` → full denoising chain for step-wise policy gradients (ReinFlow / DPPO-style)

## Gotchas

- Legacy `gym==0.22` (not Gymnasium) — pinned by the fork; `AsyncVectorEnv` uses `spawn` (MuJoCo + fork ≠ friends).
- The fork returns `obs[:2]` one step stale; `AvoidingChunkEnv._obs` rebuilds it from `env.prev_action`.
- `MUJOCO_GL=egl` requires an NVIDIA driver in the container; `set_me_up.sh` switches to `osmesa` otherwise.
  Only matters for offscreen renders — training/eval run blind.
- Python is pinned to 3.10 (D3IL/pinocchio/mujoco 3.1.6 combo); uv manages it, no system Python needed.
