# syntax=docker/dockerfile:1.7
# d3il-flow: flow-matching policies on D3IL Avoiding
#   base    - Debian + GL libs for MuJoCo (EGL / OSMesa) + uv + uv-managed CPython 3.10
#   deps    - /opt/venv with all third-party deps (project itself NOT installed -> cache-friendly)
#   dev     - devcontainer target: non-root user, just, sudo, dev extras
#   runtime - slim image with the project installed; `docker run d3il-flow:runtime --help`
#
#   docker build -f .devcontainer/Dockerfile --target dev --build-arg TORCH_EXTRA=cu128 -t d3il-flow:dev .
#   TORCH_EXTRA: cpu | cu124 | cu128 (cu128 = torch 2.8, required for Blackwell / sm_120)

ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.11

# uv binaries as a named stage: COPY --from=<image> does not expand build args, FROM does.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

# ------------------------------------------------------------------------------------------------ base
FROM debian:bookworm-slim AS base
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Europe/Berlin
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs openssh-client \
        # MuJoCo / glfw / OpenCV runtime libs; EGL for GPU hosts, OSMesa + llvmpipe for CPU-only hosts
        libgl1 libgl1-mesa-dri libglew2.2 libosmesa6 libegl1 libgles2 libglfw3 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON=${PYTHON_VERSION}
RUN uv python install ${PYTHON_VERSION}

# MuJoCo GL backend: egl needs an NVIDIA driver in the container; set_me_up.sh flips to osmesa if none.
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# ------------------------------------------------------------------------------------------------ deps
FROM base AS deps
ARG TORCH_EXTRA=cpu
ARG D3IL_REPO=https://github.com/allenzren/d3il
ARG D3IL_REF=139dbf9b114d0f6192e5433ebcffeb0fc17098f4
WORKDIR /workspace

# D3IL fork (DPPO's): avoiding-m5 env, 4-D obs, 2-D delta action. Pinned; editable path dep in pyproject.
RUN git clone --filter=blob:none ${D3IL_REPO} third_party/d3il \
    && git -C third_party/d3il checkout -q ${D3IL_REF} \
    && rm -rf third_party/d3il/.git

# Resolve + install deps only (project code is bind-mounted later); stub package so hatchling metadata is happy
COPY pyproject.toml uv.lock* ./
RUN mkdir -p src/d3il_flow && touch src/d3il_flow/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --extra ${TORCH_EXTRA} --extra dev --no-install-project; \
    else uv sync --extra ${TORCH_EXTRA} --extra dev --no-install-project; fi
ENV PATH=/opt/venv/bin:${PATH}

# ------------------------------------------------------------------------------------------------ dev
FROM deps AS dev
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends sudo less vim-tiny htop procps rsync unzip zsh \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
    && groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chown -R ${UID}:${GID} /opt/venv /opt/uv /workspace
USER ${USERNAME}
ENV HOME=/home/${USERNAME} \
    UV_CACHE_DIR=/home/${USERNAME}/.cache/uv \
    DPPO_DATA_DIR=/workspace/data \
    PYTHONUNBUFFERED=1
RUN mkdir -p ${UV_CACHE_DIR} && git config --global --add safe.directory /workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]

# ------------------------------------------------------------------------------------------------ runtime
FROM deps AS runtime
ARG TORCH_EXTRA=cpu
COPY src ./src
COPY scripts ./scripts
COPY justfile README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --extra ${TORCH_EXTRA} --no-dev
ENV DPPO_DATA_DIR=/workspace/data
ENTRYPOINT ["d3il-flow-eval"]
CMD ["--help"]
