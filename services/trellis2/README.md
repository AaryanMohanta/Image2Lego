# TRELLIS.2 image-to-3D microservice

Wraps Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
image-to-3D pipeline behind a small FastAPI service, so
`image2lego build --backend trellis2` can call it over HTTP instead of
needing TRELLIS.2's (heavy, GPU-only) dependencies installed alongside
image2lego itself.

## Requirements

- **Linux host.** TRELLIS.2 is currently only tested on Linux.
- **NVIDIA GPU with at least 24 GB of VRAM**, verified by the TRELLIS.2
  project on A100 and H100 GPUs.
- **CUDA Toolkit 12.4** (recommended by TRELLIS.2 for compiling its native
  extensions -- flash-attn, nvdiffrast, etc. -- which is also why this
  image builds from a CUDA "devel", not "runtime", base).
- NVIDIA Container Toolkit installed on the host, so Docker can pass the
  GPU through (`--gpus all`).

## Build

    docker build -t image2lego-trellis2 services/trellis2

This clones https://github.com/microsoft/TRELLIS.2 at build time and runs
its `setup.sh`, so expect the build to take a while and to need several GB
of disk. Model checkpoints are downloaded lazily on first request instead
(at container start-up cost rather than image-build cost).

## Run

    docker run --rm --gpus all -p 8765:8765 image2lego-trellis2

Then point image2lego at it (the default already matches this port):

    export TRELLIS2_URL=http://localhost:8765
    image2lego build photo.jpg --backend trellis2 --width 40 --out outdir/

## API

- `POST /generate` -- multipart form with an `image` file, optional
  `resolution` (int, default 512) and `seed` (int, default 0). Returns raw
  GLB bytes (`Content-Type: model/gltf-binary`).
- `GET /health` -- liveness check, `{"status": "ok"}`.

## Notes / things to verify before production use

This Dockerfile and app.py were written against the "Quick Start" and
"Export to GLB" sections of TRELLIS.2's own README, fetched at the time
these files were written -- not against a working build (per the task that
produced this service: it's meant to be plausible, not run-and-verified in
that session). Every call or flag beyond what the README literally showed
is marked with a `TODO` comment pointing at the specific TRELLIS.2 repo
file to check it against (the pipeline's `run()` kwargs, `to_glb()`'s full
parameter list, and the exact conda env name `setup.sh --new-env` creates,
in particular). Do that verification -- and an actual build/run on a
24 GB+ GPU host -- before relying on this for anything beyond a starting
point.
