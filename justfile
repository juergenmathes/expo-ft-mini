# d3il-flow command catalog        just --list
set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

torch := env("TORCH_EXTRA", "cpu")

default:
    @just --list --unsorted

# Idempotent workspace setup (clone D3IL fork, uv sync, GL backend, smoke test)
setup:
    bash .devcontainer/set_me_up.sh

# Re-sync the venv with pyproject (cpu | cu124 | cu128 via TORCH_EXTRA)
sync:
    uv sync --extra {{torch}} --extra dev

# Write/refresh uv.lock (commit it for reproducible builds)
lock:
    uv lock

# Download DPPO's pre-processed data for a split: just data m1|m2|m3
data mode="m1":
    bash scripts/download_data.sh {{mode}}

# One 100-step episode in avoiding-m5, prints timings + terminal info
smoke:
    uv run d3il-flow-smoke

# CFM-BC pretraining:  just train m1 --steps 30000 --hidden 512   (a leading `--` is optional)
train mode="m1" *args="":
    uv run d3il-flow-train --data data/d3il/avoid_{{mode}}/train.npz --out outputs/avoid_{{mode}}_flow {{ trim_start_match(args, "-- ") }}

# Parallel eval + rollouts.png:  just eval m1 --n-episodes 500 --n-envs 16 --flow-steps 4
eval mode="m1" *args="":
    uv run d3il-flow-eval --ckpt outputs/avoid_{{mode}}_flow/ckpt.pt {{ trim_start_match(args, "-- ") }}

# torch / CUDA sanity
gpu:
    uv run python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available(), torch.version.cuda)"

# MuJoCo offscreen render sanity (EGL on GPU hosts, OSMesa otherwise)
render:
    uv run python -c "import mujoco, os; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); r=mujoco.Renderer(m,64,64); r.update_scene(mujoco.MjData(m)); print('render ok via', os.environ.get('MUJOCO_GL'), r.render().shape)"

lint:
    uv run ruff check src
    uv run ruff format --check src

fmt:
    uv run ruff format src
    uv run ruff check --fix src

# Build a target of the Dockerfile: just docker dev|runtime
docker target="runtime":
    docker build -f .devcontainer/Dockerfile --target {{target}} --build-arg TORCH_EXTRA={{torch}} -t d3il-flow:{{target}} .

clean:
    rm -rf outputs/* .ruff_cache

# Env contract tests (no torch needed)
test:
    uv run pytest -q tests
