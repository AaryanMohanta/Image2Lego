"""Abstract image-to-mesh backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

DEFAULT_PADDING = 0.08
DEFAULT_SIZE = 1024


class ImageToMesh(ABC):
    """Turns a single product photo into a (textured/coloured) .glb mesh."""

    @abstractmethod
    def generate(self, image: Image.Image, out_dir: Path, seed: int = 0) -> Path:
        """Generate a mesh from `image`, writing it (and any
        backend-specific intermediate files) under out_dir, and returning
        the path to the resulting .glb file."""
        raise NotImplementedError

    def preprocess(
        self,
        image: Image.Image,
        padding: float = DEFAULT_PADDING,
        size: int = DEFAULT_SIZE,
    ) -> Image.Image:
        """Remove the background, crop to the subject's bounding box with
        `padding` extra on each side (as a fraction of the box's own
        width/height), pad to a square on white, and resize to size x size
        -- the input most image-to-3D backends expect.

        Needs the optional "frontend" extra (rembg) installed; imported
        lazily so importing this module (or using FileBackend, which
        doesn't call this) never requires it.
        """
        try:
            from rembg import remove  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "preprocess() needs the 'frontend' extra: install with "
                "`uv sync --extra frontend` (or `pip install image2lego[frontend]`)."
            ) from e

        removed = remove(image)
        if removed.mode != "RGBA":
            removed = removed.convert("RGBA")

        alpha = removed.split()[3]
        bbox = alpha.getbbox()
        if bbox is None:
            bbox = (0, 0, removed.width, removed.height)
        x0, y0, x1, y1 = bbox
        pad_x = round((x1 - x0) * padding)
        pad_y = round((y1 - y0) * padding)
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(removed.width, x1 + pad_x)
        y1 = min(removed.height, y1 + pad_y)
        cropped = removed.crop((x0, y0, x1, y1))

        flattened = Image.new("RGB", cropped.size, (255, 255, 255))
        flattened.paste(cropped, mask=cropped.split()[3])

        side = max(flattened.width, flattened.height)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(flattened, ((side - flattened.width) // 2, (side - flattened.height) // 2))

        return square.resize((size, size), Image.Resampling.LANCZOS)
