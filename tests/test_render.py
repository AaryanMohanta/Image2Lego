import time
from pathlib import Path

import numpy as np
from PIL import Image

from image2lego.colours import Colour
from image2lego.legolize.greedy import greedy_all
from image2lego.model import Brick, PartCatalogue
from image2lego.render import render_preview

_CATALOGUE = PartCatalogue()
_RED = Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(201, 26, 9))
_BLUE = Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0, 85, 191))


class TestRenderPreview:
    def test_empty_bricks_does_not_crash(self, tmp_path: Path) -> None:
        out = tmp_path / "preview.png"
        render_preview([], {}, out)
        assert out.exists()

    def test_single_brick_produces_non_blank_image(self, tmp_path: Path) -> None:
        out = tmp_path / "preview.png"
        bricks = [Brick(x=0, y=0, z=0, w=2, d=4, colour=4, part_id="3001")]
        render_preview(bricks, {4: _RED}, out)

        assert out.exists()
        img = np.array(Image.open(out).convert("RGB"))
        # Not every pixel is white background -- something was drawn.
        assert (img != 255).any()

    def test_unknown_colour_id_falls_back_without_crashing(self, tmp_path: Path) -> None:
        out = tmp_path / "preview.png"
        bricks = [Brick(x=0, y=0, z=0, w=1, d=1, colour=999, part_id="3005")]
        render_preview(bricks, {4: _RED}, out)
        assert out.exists()

    def test_plate_layer_type_uses_plate_height(self, tmp_path: Path) -> None:
        out_brick = tmp_path / "brick.png"
        out_plate = tmp_path / "plate.png"
        bricks = [Brick(x=0, y=0, z=0, w=2, d=4, colour=4, part_id="3020")]

        render_preview(bricks, {4: _RED}, out_brick, layer_type="brick")
        render_preview(bricks, {4: _RED}, out_plate, layer_type="plate")

        # A shorter (plate) brick should produce a visibly smaller/different
        # silhouette than a taller (brick) one -- images should differ.
        brick_img = np.array(Image.open(out_brick).convert("RGB"))
        plate_img = np.array(Image.open(out_plate).convert("RGB"))
        assert not np.array_equal(brick_img, plate_img)

    def test_renders_a_realistic_model_reasonably_fast(self, tmp_path: Path) -> None:
        colours = np.full((10, 10, 10), -1, dtype=np.int16)
        colours[:, :, :] = 4
        colours[0:5, :, :] = 1
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)
        assert len(bricks) > 20

        out = tmp_path / "preview.png"
        start = time.perf_counter()
        render_preview(bricks, {4: _RED, 1: _BLUE}, out)
        elapsed = time.perf_counter() - start

        assert out.exists()
        assert elapsed < 10.0
