"""Trellis2Backend: talks to a local HTTP microservice running Microsoft's
TRELLIS.2 image-to-3D model.

TRELLIS.2 itself needs a Linux host with an NVIDIA GPU with >= 24 GB VRAM,
so it isn't something this library runs in-process -- see
services/trellis2/ (Dockerfile + FastAPI app) for the microservice this
backend is a thin HTTP client for, and its README for how to build/run it.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import requests
from PIL import Image

from image2lego.frontend.base import ImageToMesh

logger = logging.getLogger(__name__)

ENV_TRELLIS2_URL = "TRELLIS2_URL"
DEFAULT_TRELLIS2_URL = "http://localhost:8765"


class Trellis2Backend(ImageToMesh):
    def __init__(
        self,
        base_url: str | None = None,
        resolution: int = 512,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get(ENV_TRELLIS2_URL, DEFAULT_TRELLIS2_URL)
        ).rstrip("/")
        self.resolution = resolution
        self.timeout = timeout

    def generate(self, image: Image.Image, out_dir: Path, seed: int = 0) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        processed = self.preprocess(image)

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)

        url = f"{self.base_url}/generate"
        logger.info(
            "requesting mesh from %s (resolution=%d, seed=%d)", url, self.resolution, seed
        )
        response = requests.post(
            url,
            files={"image": ("subject.png", buf, "image/png")},
            data={"resolution": str(self.resolution), "seed": str(seed)},
            timeout=self.timeout,
        )
        response.raise_for_status()

        dest = out_dir / "mesh.glb"
        dest.write_bytes(response.content)
        return dest
