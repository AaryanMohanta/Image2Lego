"""FastAPI microservice wrapping TRELLIS.2's image-to-3D pipeline.

Run inside the Docker image built from this directory's Dockerfile, on a
Linux host with an NVIDIA GPU with >= 24 GB VRAM (see README.md). Exposes:

    POST /generate  (multipart: image=<file>, resolution=<int>, seed=<int>)
        -> raw GLB bytes (Content-Type: model/gltf-binary)
    GET  /health -> {"status": "ok"}

Grounded against the "Quick Start: Image-to-3D Generation" and "Export to
GLB" sections of https://github.com/microsoft/TRELLIS.2's README, as
fetched when this file was written:

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    image = Image.open("image.png")
    mesh = pipeline.run(image)[0]

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
    )
    glb.export("output.glb", extension_webp=True)

Anything below that goes beyond that literal snippet (a `seed`/`resolution`
kwarg on .run(), exact conda env name, etc.) is marked with a TODO pointing
at the exact repo file to verify it against before relying on this in
production.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image

logger = logging.getLogger("trellis2")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TRELLIS.2 image-to-3D microservice")

_pipeline: Any = None


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        # README "Quick Start: Image-to-3D Generation".
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # type: ignore[import-not-found]

        logger.info("loading TRELLIS.2-4B pipeline (first request only)...")
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _pipeline.cuda()
    return _pipeline


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    resolution: int = Form(512),
    seed: int = Form(0),
) -> Response:
    pipeline = _get_pipeline()

    raw = await image.read()
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")

    logger.info("running TRELLIS.2: resolution=%d seed=%d", resolution, seed)

    # TODO verify pipeline.run()'s exact kwargs against
    # https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py
    # (or wherever Trellis2ImageTo3DPipeline.run is defined) -- the
    # README's quick-start only shows `pipeline.run(image)` with no
    # arguments; `seed` and `resolution` kwargs are assumed here by
    # analogy with similar diffusion pipelines, not confirmed against
    # TRELLIS.2's own signature. If either isn't accepted, drop it (seed
    # in particular may need to be set globally, e.g. via
    # torch.manual_seed(seed), rather than passed to run()).
    mesh = pipeline.run(pil_image, seed=seed, resolution=resolution)[0]

    import o_voxel  # type: ignore[import-not-found]  # README "Export to GLB"

    # TODO verify against
    # https://github.com/microsoft/TRELLIS.2/blob/main/o_voxel/postprocess.py
    # -- to_glb()'s exact parameter list and defaults; decimation_target
    # and texture_size below are the README example's literal values
    # (texture_size swapped for the request's `resolution` here).
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000,
        texture_size=resolution,
    )

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "output.glb"
        # README's example passes extension_webp=True to .export(), not
        # to_glb() -- kept here to match that literal example.
        glb.export(str(out_path), extension_webp=True)
        glb_bytes = out_path.read_bytes()

    return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
