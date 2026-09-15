import logging

import numpy as np
import pytest

from image2lego.legolize.graph import build_graph, grounded_components, weak_points
from image2lego.legolize.repair import (
    dissolve,
    expand_region_to_cover_intersecting_bricks,
    ground_floating,
    repair_weak_points,
)
from image2lego.model import Brick, PartCatalogue

_CATALOGUE = PartCatalogue()


def _b(x: int, y: int, z: int, w: int, d: int, colour: int = 4, part_id: str = "3024") -> Brick:
    return Brick(x=x, y=y, z=z, w=w, d=d, colour=colour, part_id=part_id)


class TestDissolve:
    def test_removes_bricks_fully_inside_region(self) -> None:
        bricks = [_b(1, 0, 1, 1, 1), _b(5, 0, 5, 1, 1)]
        remaining, _grid = dissolve(bricks, region=(0, 1, 0, 3, 0, 3))
        assert remaining == [bricks[1]]

    def test_removes_bricks_that_only_partially_overlap_region(self) -> None:
        bricks = [_b(2, 0, 0, 2, 2)]  # spans x=2..4, partially inside x<3 region
        remaining, _grid = dissolve(bricks, region=(0, 1, 0, 3, 0, 3))
        assert remaining == []

    def test_colour_subgrid_shape_matches_region(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1)]
        _remaining, grid = dissolve(bricks, region=(0, 2, 0, 3, 0, 4))
        assert grid.shape == (3, 2, 4)

    def test_colour_subgrid_has_removed_bricks_colours(self) -> None:
        bricks = [_b(1, 0, 1, 1, 1, colour=7)]
        _remaining, grid = dissolve(bricks, region=(0, 1, 0, 3, 0, 3))
        assert grid[1, 0, 1] == 7

    def test_empty_region_returns_all_bricks_unchanged(self) -> None:
        bricks = [_b(5, 0, 5, 1, 1)]
        remaining, grid = dissolve(bricks, region=(0, 1, 0, 2, 0, 2))
        assert remaining == bricks
        assert (grid == -1).all()


class TestExpandRegionToCoverIntersectingBricks:
    def test_fully_contains_a_long_chain_of_overlapping_bricks(self) -> None:
        # 12 bricks in a row, each width 2 and offset by 1 from the last
        # (brick i spans x=[i, i+2)), so consecutive bricks overlap by
        # exactly one cell. Because the touched-bricks scan only sees
        # overlaps against *this round's* current bounds, absorbing this
        # chain from a nominal window on the last brick back to the first
        # takes 11 rounds of growth -- more than the function's old fixed
        # cap of 8, which silently returned a region that still had
        # bricks 0-2 poking outside it (a real bug: dissolve() would then
        # drop the outside-the-region portion of their colour data,
        # observed in practice on a 48-wide model's densely tiled brick
        # rows).
        bricks = [Brick(x=i, y=0, z=0, w=2, d=1, colour=4, part_id="3023") for i in range(12)]
        nominal = (0, 1, 11, 13, 0, 1)

        region = expand_region_to_cover_intersecting_bricks(bricks, nominal)
        y0, y1, x0, x1, z0, z1 = region

        for brick in bricks:
            intersects = (
                y0 <= brick.y < y1
                and brick.x < x1
                and brick.x + brick.w > x0
                and brick.z < z1
                and brick.z + brick.d > z0
            )
            if intersects:
                assert brick.x >= x0 and brick.x + brick.w <= x1, (
                    f"{brick} intersects the region but isn't fully contained by it"
                )


class TestGroundFloating:
    def _catalogue(self) -> PartCatalogue:
        return _CATALOGUE

    def test_grounds_floating_component_through_solid_interior(self) -> None:
        # A 1x1 floating at y=2, with an empty (uncoloured) but solid column
        # beneath it all the way to the ground.
        bricks = [_b(0, 2, 0, 1, 1)]
        colours = np.full((1, 3, 1), -1, dtype=np.int16)
        colours[0, 2, 0] = 4
        solid_occ = np.ones((1, 3, 1), dtype=bool)

        result = ground_floating(bricks, colours, solid_occ, _CATALOGUE, "brick")

        g = build_graph(result)
        grounded, floating = grounded_components(g, result)
        assert floating == []
        assert len(result) == 3  # original + 2 filler layers (y=1, y=0)

    def test_drops_component_with_no_path_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bricks = [_b(0, 2, 0, 1, 1)]
        colours = np.full((1, 3, 1), -1, dtype=np.int16)
        colours[0, 2, 0] = 4
        solid_occ = np.zeros((1, 3, 1), dtype=bool)
        solid_occ[0, 2, 0] = True  # only the floating brick's own cell is solid

        with caplog.at_level(logging.WARNING):
            result = ground_floating(bricks, colours, solid_occ, _CATALOGUE, "brick")

        assert result == []
        assert "floating" in caplog.text.lower()

    def test_grounded_bricks_pass_through_unchanged(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1)]
        colours = np.full((1, 1, 1), 4, dtype=np.int16)
        solid_occ = np.ones((1, 1, 1), dtype=bool)

        result = ground_floating(bricks, colours, solid_occ, _CATALOGUE, "brick")
        assert result == bricks

    def test_grounds_stacked_floating_components_regardless_of_processing_order(
        self,
    ) -> None:
        # Two separate floating components, one stacked above the other in
        # the same (x, z) column, both reachable via solid (uncoloured)
        # routing space -- but the upper one's straight-down path passes
        # directly through the lower one's own footprint cell. Grounding
        # the lower component first unblocks the upper one; grounding the
        # upper one first (before the lower one is in place) makes its
        # vertical column collide with the still-floating lower brick and
        # fail. Since nx.connected_components' traversal order follows
        # node/brick-list insertion order, listing the upper component's
        # brick BEFORE the lower one's forces the "wrong" order on a
        # single linear pass.
        upper = _b(0, 4, 0, 1, 1)  # floating, lowest_y=4
        lower = _b(0, 2, 0, 1, 1)  # floating, lowest_y=2
        base = _b(0, 0, 0, 1, 1)  # grounded, y=0
        bricks = [upper, lower, base]

        colours = np.full((1, 5, 1), -1, dtype=np.int16)
        colours[0, 0, 0] = 4
        colours[0, 2, 0] = 4
        colours[0, 4, 0] = 4
        solid_occ = np.ones((1, 5, 1), dtype=bool)

        result = ground_floating(bricks, colours, solid_occ, _CATALOGUE, "brick")

        covered = np.zeros(colours.shape, dtype=np.int32)
        for b in result:
            for x, z in b.footprint:
                covered[x, b.y, z] += 1
        assert np.all(covered[colours != -1] == 1)

        g = build_graph(result)
        grounded, floating = grounded_components(g, result)
        assert floating == []


class TestRepairWeakPoints:
    def test_no_weak_points_returns_bricks_unchanged(self) -> None:
        bricks = [_b(0, 0, 0, 4, 4)]
        colours = np.full((4, 1, 4), 4, dtype=np.int16)
        solid_occ = np.ones((4, 1, 4), dtype=bool)
        rng = np.random.default_rng(0)

        result = repair_weak_points(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=10
        )
        assert result == bricks

    def test_does_not_break_coverage_invariant(self) -> None:
        # A bridge-brick structure with an articulation point to repair.
        bricks = [
            _b(0, 0, 0, 4, 4, colour=4),
            _b(1, 1, 1, 1, 1, colour=4),
            _b(0, 2, 0, 4, 4, colour=4),
        ]
        colours = np.full((4, 3, 4), -1, dtype=np.int16)
        colours[0:4, 0, 0:4] = 4
        colours[1, 1, 1] = 4
        colours[0:4, 2, 0:4] = 4
        solid_occ = colours != -1
        rng = np.random.default_rng(0)

        result = repair_weak_points(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=20
        )

        covered = np.zeros(colours.shape, dtype=np.int32)
        for b in result:
            for x, z in b.footprint:
                covered[x, b.y, z] += 1
        np.testing.assert_array_equal(covered > 0, colours != -1)
        assert (covered <= 1).all()

    def test_reduces_or_maintains_articulation_point_count(self) -> None:
        bricks = [
            _b(0, 0, 0, 4, 4, colour=4),
            _b(1, 1, 1, 1, 1, colour=4),
            _b(0, 2, 0, 4, 4, colour=4),
        ]
        colours = np.full((4, 3, 4), -1, dtype=np.int16)
        colours[0:4, 0, 0:4] = 4
        colours[1, 1, 1] = 4
        colours[0:4, 2, 0:4] = 4
        solid_occ = colours != -1
        rng = np.random.default_rng(0)

        before_graph = build_graph(bricks)
        before_articulation = len(weak_points(before_graph, bricks)["articulation_points"])

        result = repair_weak_points(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=50
        )
        after_graph = build_graph(result)
        after_articulation = len(weak_points(after_graph, result)["articulation_points"])

        assert after_articulation <= before_articulation
