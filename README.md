# expo-ft-mini

Flow-matching BC on D3IL Avoiding. Tiny MLP, 8 demos (`m1`), ~95% success on the default maze.

## 1. Container

```bash
git clone https://github.com/juergenmathes/expo-ft-mini.git && cd expo-ft-mini
export TORCH_EXTRA=cu128    # Blackwell (RTX 50 / PRO 2000). Use cu124 on Ampere/Ada, omit on CPU.
```

Cursor / VS Code: **Reopen in Container**. `postCreate` clones D3IL, syncs the venv, smokes one episode.

CLI instead:

```bash
just docker dev
docker run -it --rm --gpus all -v $PWD:/workspace d3il-flow:dev bash
just setup
```

## 2. Train + eval (what we ran)

```bash
just gpu                       # expect: 2.8.0+cu128 cuda: True 12.8
just data m1
just train m1                  # ~80s on GPU → outputs/avoid_m1_flow/ckpt.pt
just eval m1 --device cuda --n-episodes 200 --n-envs 16
```

Should print **~0.95 success**. Plot: `outputs/avoid_m1_flow/rollouts.png`.

Harder maze, **same frozen ckpt** (fatter last-row pillars → ~0.36 success):

```bash
AVOID_L3_RADIUS=0.0575 just eval m1 --device cuda --n-episodes 200 --n-envs 16
```

`AVOID_OFFSET` (default `0.075`) moves the columns. Unset = original maze.
