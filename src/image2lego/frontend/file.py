"""FileBackend: returns a fixed mesh path, ignoring the input image.

For development and tests, where running a real image-to-3D backend isn't
practical or (for a fixed test mesh) even desired.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from image2lego.frontend.base import ImageToMesh


class FileBackend(ImageToMesh):
    def __init__(self, mesh_path: str | Path) -> None:
        self.mesh_path = Path(mesh_path)

    def generate(self, image: Image.Image, out_dir: Path, seed: int = 0) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "mesh.glb"
        shutil.copy(self.mesh_path, dest)
        return dest
