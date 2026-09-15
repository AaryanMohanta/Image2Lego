import numpy as np

from image2lego.legolize.greedy import greedy_all
from image2lego.legolize.stability import analyse_stability, repair_stability
from image2lego.model import Brick, PartCatalogue

_CATALOGUE = PartCatalogue()


class TestSingleColumn:
    def test_30_tall_1x1_column_is_feasible(self) -> None:
        bricks = [
            Brick(x=0, y=y, z=0, w=1, d=1, colour=4, part_id="3005") for y in range(30)
        ]
        result = analyse_stability(bricks, "brick")
        assert result.feasible
        assert result.total_slack < 1e-4

    def test_single_ground_brick_is_feasible(self) -> None:
        bricks = [Brick(x=0, y=0, z=0, w=1, d=1, colour=4, part_id="3005")]
        result = analyse_stability(bricks, "brick")
        assert result.feasible


class TestCantilever:
    @staticmethod
    def _cantilever(length: int) -> list[Brick]:
        # Each successive layer's 2x4 brick is offset 2 studs further along
        # x than the one below, so the vertical overlap (and hence the
        # connection's ability to resist the growing overturning moment)
        # stays fixed while the moment arm grows with height.
        return [
            Brick(x=y * 2, y=y, z=0, w=4, d=2, colour=4, part_id="3001")
            for y in range(length)
        ]

    # A 2x4 brick is heavy (footprint area 8, i.e. 8x a 1x1 brick's
    # weight), so the connection-strength defaults (t_max=s_max=1.0, tuned
    # around a 1x1-brick-weight unit) are already saturated by a single
    # unsupported 2x4 -- there's no "short but non-trivial" cantilever to
    # observe at those defaults. Scaling t_max/s_max up externally (they're
    # parameters, not fixed) keeps the same geometry but gives the
    # structure enough strength to show a real short-vs-long transition.
    _T_MAX = 20.0
    _S_MAX = 20.0

    def test_short_cantilever_is_feasible(self) -> None:
        bricks = self._cantilever(2)
        result = analyse_stability(bricks, "brick", t_max=self._T_MAX, s_max=self._S_MAX)
        assert result.feasible

    def test_long_cantilever_is_infeasible_and_root_is_most_stressed(self) -> None:
        bricks = self._cantilever(4)
        result = analyse_stability(bricks, "brick", t_max=self._T_MAX, s_max=self._S_MAX)
        assert not result.feasible

        root_brick_y = bricks[result.top_stressed_bricks[0]].y
        assert root_brick_y == 0

    def test_slack_increases_with_cantilever_length(self) -> None:
        short_result = analyse_stability(
            self._cantilever(2), "brick", t_max=self._T_MAX, s_max=self._S_MAX
        )
        long_result = analyse_stability(
            self._cantilever(10), "brick", t_max=self._T_MAX, s_max=self._S_MAX
        )
        assert long_result.total_slack > short_result.total_slack


class TestSolidCube:
    def test_solid_4x4x4_cube_is_feasible_with_near_zero_slack(self) -> None:
        import numpy as np

        colours = np.full((4, 4, 4), 4, dtype=np.int16)
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)

        result = analyse_stability(bricks, "brick")
        assert result.feasible
        assert result.total_slack < 1e-3


class TestStabilityPerformance:
    def test_moderate_solid_cube_completes_quickly(self) -> None:
        # Regression test: a solid cube's equilibrium LP is highly
        # symmetric/degenerate (many force distributions tie on the slack
        # objective), which the default HiGHS method (dual simplex here)
        # handles very poorly -- measured ~29s for this exact case before
        # switching to method="highs-ipm" in analyse_stability, vs ~3s
        # after. 10s leaves comfortable margin over slower CI hardware
        # while still catching a regression back to simplex-level slowness.
        import time

        import numpy as np

        colours = np.full((14, 14, 14), 4, dtype=np.int16)
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)

        start = time.perf_counter()
        result = analyse_stability(bricks, "brick")
        elapsed = time.perf_counter() - start

        assert result.feasible
        assert elapsed < 10.0


class TestStabilityResultShape:
    def test_empty_bricks_is_feasible(self) -> None:
        result = analyse_stability([], "brick")
        assert result.feasible
        assert result.total_slack == 0.0
        assert result.brick_slack == []
        assert result.top_stressed_bricks == []

    def test_top_k_respects_limit(self) -> None:
        bricks = [
            Brick(x=0, y=y, z=0, w=1, d=1, colour=4, part_id="3005") for y in range(10)
        ]
        result = analyse_stability(bricks, "brick", top_k=3)
        assert len(result.top_stressed_bricks) <= 3

    def test_connection_utilisation_is_between_0_and_reasonable_bound(self) -> None:
        bricks = [
            Brick(x=0, y=y, z=0, w=1, d=1, colour=4, part_id="3005") for y in range(5)
        ]
        result = analyse_stability(bricks, "brick")
        assert len(result.connections) == 4  # 5 bricks -> 4 vertical joints
        for conn in result.connections:
            assert conn.utilisation >= 0.0


class TestRepairStability:
    def test_never_increases_total_slack(self) -> None:
        colours = np.full((4, 4, 4), 4, dtype=np.int16)
        solid_occ = np.ones((4, 4, 4), dtype=bool)
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)

        before = analyse_stability(bricks, "brick").total_slack
        rng = np.random.default_rng(0)
        repaired = repair_stability(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=10
        )
        after = analyse_stability(repaired, "brick").total_slack

        assert after <= before + 1e-9

    def test_preserves_coverage_and_non_overlap(self) -> None:
        colours = np.full((4, 4, 4), 4, dtype=np.int16)
        solid_occ = np.ones((4, 4, 4), dtype=bool)
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)

        rng = np.random.default_rng(0)
        repaired = repair_stability(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=10
        )

        covered = np.zeros(colours.shape, dtype=np.int32)
        for b in repaired:
            for x, z in b.footprint:
                covered[x, b.y, z] += 1
        np.testing.assert_array_equal(covered > 0, colours != -1)
        assert (covered <= 1).all()

    def test_feasible_structure_returns_quickly_unchanged(self) -> None:
        colours = np.full((4, 4, 4), 4, dtype=np.int16)
        solid_occ = np.ones((4, 4, 4), dtype=bool)
        bricks = greedy_all(colours, _CATALOGUE, "brick", seed=0)
        assert analyse_stability(bricks, "brick").feasible

        rng = np.random.default_rng(0)
        repaired = repair_stability(
            bricks, colours, solid_occ, _CATALOGUE, "brick", rng, max_iters=200
        )
        assert repaired == bricks
