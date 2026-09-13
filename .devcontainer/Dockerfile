# syntax=docker/dockerfile:1.7
# Image = vscode Debian + MuJoCo GL libs + uv + CPython 3.10.
# The project venv is NOT in the image: postCreate/`just setup` runs `uv sync` → /workspace/.venv
#
#   docker build -f .devcontainer/Dockerfile --target dev -t d3il-flow:dev .
#   docker build -f .devcontainer/Dockerfile --target runtime --build-arg TORCH_EXTRA=cu128 -t d3il-flow:runtime .

ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.11

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uvbin

FROM mcr.microsoft.com/devcontainers/base:bookworm AS base
ARG PYTHON_VERSION
USER root
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Europe/Berlin
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgl1-mesa-dri libglew2.2 libosmesa6 libegl1 libgles2 libglfw3 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 libxi6 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /uvx /usr/local/bin/
# Interpreter only (world-readable). Project packages live in the workspace .venv.
ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON=${PYTHON_VERSION}
RUN uv python install ${PYTHON_VERSION}

# egl needs an NVIDIA driver; set_me_up.sh flips to osmesa if none.
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

FROM base AS dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
USER vscode
ENV HOME=/home/vscode \
    UV_CACHE_DIR=/home/vscode/.cache/uv \
    DPPO_DATA_DIR=/workspace/data \
    PYTHONUNBUFFERED=1
RUN mkdir -p ${UV_CACHE_DIR} && git config --global --add safe.directory /workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]

FROM base AS runtime
ARG TORCH_EXTRA=cpu
ARG D3IL_REPO=https://github.com/allenzren/d3il
ARG D3IL_REF=139dbf9b114d0f6192e5433ebcffeb0fc17098f4
WORKDIR /workspace
RUN git clone --filter=blob:none ${D3IL_REPO} third_party/d3il \
    && git -C third_party/d3il checkout -q ${D3IL_REF} \
    && rm -rf third_party/d3il/.git
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY scripts ./scripts
COPY justfile README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --extra ${TORCH_EXTRA} --no-dev; \
    else uv sync --extra ${TORCH_EXTRA} --no-dev; fi
ENV DPPO_DATA_DIR=/workspace/data \
    PATH=/workspace/.venv/bin:${PATH}
ENTRYPOINT ["d3il-flow-eval"]
CMD ["--help"]
