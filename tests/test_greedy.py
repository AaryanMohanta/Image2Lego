import time

import numpy as np

from image2lego.legolize.greedy import _place_largest_containing, greedy_all, greedy_layer
from image2lego.model import Brick, PartCatalogue

_CATALOGUE = PartCatalogue()


def _assert_covers_exactly(bricks: list[Brick], colour_layer: np.ndarray, y: int) -> None:
    covered = np.zeros(colour_layer.shape, dtype=np.int32)
    for b in bricks:
        assert b.y == y
        for x, z in b.footprint:
            covered[x, z] += 1
    expected = colour_layer != -1
    np.testing.assert_array_equal(covered > 0, expected)
    assert (covered <= 1).all()


class TestGreedyLayer:
    def test_covers_uniform_layer_with_few_large_bricks(self) -> None:
        layer = np.full((8, 8), 4, dtype=np.int16)
        rng = np.random.default_rng(0)
        bricks = greedy_layer(layer, y=2, catalogue=_CATALOGUE, layer_type="brick", rng=rng)

        _assert_covers_exactly(bricks, layer, y=2)
        assert len(bricks) < 20  # far fewer than 64 1x1s

    def test_never_crosses_colour_boundary(self) -> None:
        layer = np.zeros((6, 6), dtype=np.int16)
        layer[:, :3] = 1
        layer[:, 3:] = 2
        rng = np.random.default_rng(1)
        bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng)

        _assert_covers_exactly(bricks, layer, y=0)
        for b in bricks:
            for x, z in b.footprint:
                assert layer[x, z] == b.colour

    def test_skips_empty_cells(self) -> None:
        layer = np.full((5, 5), 3, dtype=np.int16)
        layer[2, 2] = -1
        rng = np.random.default_rng(2)
        bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng)

        _assert_covers_exactly(bricks, layer, y=0)
        for b in bricks:
            assert (2, 2) not in b.footprint

    def test_plate_layer_type_uses_plate_part_ids(self) -> None:
        layer = np.full((2, 4), 5, dtype=np.int16)
        rng = np.random.default_rng(3)
        bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="plate", rng=rng)
        assert len(bricks) == 1
        assert bricks[0].part_id == "3020"  # plate id for the 2x4 size class

    def test_random_irregular_blob_has_exact_coverage(self) -> None:
        rng_data = np.random.default_rng(42)
        layer = np.full((10, 10), -1, dtype=np.int16)
        mask = rng_data.random((10, 10)) < 0.6
        layer[mask] = rng_data.integers(0, 3, size=mask.sum())

        rng = np.random.default_rng(4)
        bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng)
        _assert_covers_exactly(bricks, layer, y=0)

    def test_60x60_layer_completes_in_well_under_a_second(self) -> None:
        rng_data = np.random.default_rng(0)
        layer = rng_data.integers(0, 4, size=(60, 60)).astype(np.int16)

        rng = np.random.default_rng(5)
        start = time.perf_counter()
        bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng)
        elapsed = time.perf_counter() - start

        _assert_covers_exactly(bricks, layer, y=0)
        assert elapsed < 1.0

    def test_seed_cell_places_a_bridging_brick_first(self) -> None:
        # A 2-deep base rectangle (z=0..1) with a 1-wide notch reaching
        # z=2 only at x=1..2. Left to its own devices, greedy_layer's
        # largest-rectangle-first scan places a (w=4, d=2) brick over the
        # whole base *before* ever considering the notch, stranding the
        # notch as its own separate (2, 1) sliver with no footprint
        # overlap with anything at z<2 -- structurally disconnected from
        # whatever sits below z=0..1 in an adjacent layer. seed_cell
        # should force the *first* placement to be the largest brick
        # containing that cell, which here is the one shape that spans
        # z=0..2 together (bridging the notch to the base).
        layer = np.array(
            [
                [4, 4, -1],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, -1],
            ],
            dtype=np.int16,
        )
        rng = np.random.default_rng(0)

        bricks = greedy_layer(
            layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng, seed_cell=(1, 2)
        )

        _assert_covers_exactly(bricks, layer, y=0)
        bridging = [b for b in bricks if (1, 2) in b.footprint]
        assert len(bridging) == 1
        assert any(z < 2 for _x, z in bridging[0].footprint)

    def test_seed_bias_cell_prefers_the_closer_valid_bridging_offset(self) -> None:
        # A 3-wide notch (x=2..4, z=2) too narrow for any brick bigger
        # than (2, 3) to bridge, so exactly two same-sized placements
        # validly bridge z=0..2: one anchored at x=2, one at x=3. Without
        # a bias, a uniform-random pick between them can land on either;
        # with seed_bias_cell pointing toward the far (x=6) side, the
        # offset closer to it (anchored at x=3) should win
        # deterministically.
        layer = np.array(
            [
                [4, 4, -1],
                [4, 4, -1],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, -1],
                [4, 4, -1],
            ],
            dtype=np.int16,
        )
        for seed in range(10):
            rng = np.random.default_rng(seed)
            bricks = greedy_layer(
                layer,
                y=0,
                catalogue=_CATALOGUE,
                layer_type="brick",
                rng=rng,
                seed_cell=(3, 2),
                seed_bias_cell=(6, 0),
            )
            bridging = [b for b in bricks if (3, 2) in b.footprint]
            assert len(bridging) == 1
            assert bridging[0].x == 3

    def test_seed_bias_cell_prefers_a_smaller_bridging_size_over_a_bigger_non_bridging_one(
        self,
    ) -> None:
        # A 7-wide (x=0..6), 10-deep (z=0..9) uniformly-coloured area.
        # seed_cell and bias_cell are 2 apart in x (x=6 and x=4). The
        # single largest catalogue size, (2, 8) (area 16), has two
        # orientations: as (w=2, d=8) its only valid placement containing
        # (6, 4) is x=5..6 -- 2 cells wide, so it can never reach x=4,
        # 2 away; as (w=8, d=2) it doesn't fit in a 7-wide grid at all.
        # So neither orientation of the largest size bridges the gap, even
        # though a *smaller* size -- (2, 6) rotated to (w=6, d=2), area 12
        # -- both fits and bridges it (x=1..6 spans both cells). A version
        # of this function that stops at the first size with *any* valid
        # placement (picking the closest-to-bias option within it, as an
        # earlier version of this function did) would settle for the
        # non-bridging (2, 8) placement and never even look at (2, 6).
        layer = np.full((7, 10), 4, dtype=np.int16)
        covered = layer < 0

        for seed in range(10):
            rng = np.random.default_rng(seed)
            brick = _place_largest_containing(
                layer, covered.copy(), 6, 4, _CATALOGUE, "brick", rng, bias_cell=(4, 4)
            )
            assert brick is not None
            assert (6, 4) in brick.footprint
            assert (4, 4) in brick.footprint

    def test_seed_cell_none_matches_previous_behaviour(self) -> None:
        layer = np.full((8, 8), 4, dtype=np.int16)
        rng = np.random.default_rng(0)
        bricks = greedy_layer(
            layer, y=2, catalogue=_CATALOGUE, layer_type="brick", rng=rng, seed_cell=None
        )
        _assert_covers_exactly(bricks, layer, y=2)

    def test_different_rng_seeds_can_pick_different_corners(self) -> None:
        layer = np.full((6, 6), 1, dtype=np.int16)
        seen_first_bricks = set()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            bricks = greedy_layer(layer, y=0, catalogue=_CATALOGUE, layer_type="brick", rng=rng)
            seen_first_bricks.add((bricks[0].x, bricks[0].z))
        assert len(seen_first_bricks) > 1


class TestGreedyAll:
    def test_produces_bricks_for_every_occupied_layer(self) -> None:
        colours = np.full((4, 3, 4), -1, dtype=np.int16)
        colours[:, 0, :] = 5
        colours[:, 1, :] = 5
        colours[1:3, 2, 1:3] = 5

        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)

        covered = np.zeros(colours.shape, dtype=np.int32)
        for b in bricks:
            for x, z in b.footprint:
                covered[x, b.y, z] += 1
        np.testing.assert_array_equal(covered > 0, colours != -1)
        assert (covered <= 1).all()
