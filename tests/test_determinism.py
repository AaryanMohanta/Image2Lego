import numpy as np

from image2lego.legolize import legolize
from image2lego.model import Brick, PartCatalogue

_CATALOGUE = PartCatalogue()


def _lumpy_colours() -> tuple[np.ndarray, np.ndarray]:
    """An asymmetric, multi-colour, partially-hollow shape -- large and
    irregular enough that greedy's randomized scan corner/orientation
    choices and repair's randomized weak-point selection have real room
    to diverge between seeds, unlike a solid cube or a small/simple
    blob (empirically, small shapes have so few valid merges that
    different seeds still converge on the same layout)."""
    colours = np.full((12, 9, 14), 4, dtype=np.int16)
    colours[8:, 5:, 10:] = 14
    colours[:5, :3, :4] = 1
    colours[0, 0, :6] = -1
    colours[6, 4, :] = -1
    solid_occ = colours != -1
    return colours, solid_occ


_BrickKey = tuple[int, int, int, int, int, int, str]


def _brick_key(brick: Brick) -> _BrickKey:
    return (brick.x, brick.y, brick.z, brick.w, brick.d, brick.colour, brick.part_id)


def _sorted_keys(bricks: list[Brick]) -> list[_BrickKey]:
    return sorted(_brick_key(b) for b in bricks)


class TestDeterminism:
    def test_same_seed_produces_identical_brick_lists(self) -> None:
        colours, solid_occ = _lumpy_colours()

        bricks_a = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=7, restarts=3)
        bricks_b = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=7, restarts=3)

        assert _sorted_keys(bricks_a) == _sorted_keys(bricks_b)

    def test_different_seeds_produce_different_brick_lists(self) -> None:
        colours, solid_occ = _lumpy_colours()

        bricks_a = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=1, restarts=3)
        bricks_b = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=2, restarts=3)

        assert _sorted_keys(bricks_a) != _sorted_keys(bricks_b)
