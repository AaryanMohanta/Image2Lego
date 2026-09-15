# TRELLIS.2 image-to-3D microservice.
#
# Requirements (per https://github.com/microsoft/TRELLIS.2's README, as
# fetched when this file was written -- reconfirm against the live README
# before relying on this for anything beyond a starting point):
#   - Linux host (TRELLIS.2 is currently only tested there).
#   - NVIDIA GPU with at least 24 GB VRAM (verified on A100/H100).
#   - CUDA Toolkit 12.4 recommended for compiling TRELLIS.2's native
#     extensions (flash-attn, nvdiffrast, ...), hence the "devel" (not
#     "runtime") base image below.
#
# See README.md in this directory for how to build and run this image, and
# for the /generate API it exposes.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# System Python 3.10, per this image's own base spec. TRELLIS.2's own
# setup.sh (below) manages its actual runtime dependencies inside a conda
# env it creates itself (the --new-env flag), which may end up on a
# different Python version -- see setup.sh in the repo for exactly which,
# and adjust the `conda activate` target below (currently guessed as
# "trellis2") to match its real env name.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda: TRELLIS.2's setup.sh is a conda-based installer (its README's
# "Requirements" section lists Conda for dependency management).
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh \
    && conda init bash

WORKDIR /opt
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive trellis2
WORKDIR /opt/trellis2

# TODO verify these setup.sh flags are still current against
# https://github.com/microsoft/TRELLIS.2/blob/main/setup.sh and the
# README's "Installation" section -- this is the exact command shown in
# the README at the time this Dockerfile was written; the script may have
# added, renamed, or reordered flags since. In particular, confirm here
# whether --new-env creates a conda env literally named "trellis2" (this
# Dockerfile assumes so below) or something else.
RUN bash -lc ". ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"

# The service layer: FastAPI + what app.py needs beyond what setup.sh
# already installed into the conda env it created.
RUN bash -lc "conda activate trellis2 && pip install --no-cache-dir fastapi 'uvicorn[standard]' python-multipart"

COPY app.py /opt/trellis2/app.py

EXPOSE 8765

CMD ["bash", "-lc", "conda activate trellis2 && python -m uvicorn app:app --host 0.0.0.0 --port 8765"]
