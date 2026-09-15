import logging

import numpy as np
import pytest
import trimesh

from image2lego.geometry.voxelize import voxelize
from image2lego.legolize import legolize
from image2lego.legolize.graph import build_graph, grounded_components
from image2lego.legolize.greedy import greedy_all
from image2lego.model import Brick, PartCatalogue

_CATALOGUE = PartCatalogue()


def _solid_cube(size: int, colour: int = 4) -> tuple[np.ndarray, np.ndarray]:
    colours = np.full((size, size, size), colour, dtype=np.int16)
    solid_occ = np.ones((size, size, size), dtype=bool)
    return colours, solid_occ


def _assert_invariants(bricks: list[Brick], colours: np.ndarray) -> None:
    # Filler bricks from grounding legitimately occupy colours == -1 cells
    # (hollow-but-solid routing space), so coverage is one-way: every
    # *coloured* voxel must be covered exactly once. Overlap is checked
    # everywhere, filler included.
    covered = np.zeros(colours.shape, dtype=np.int32)
    for b in bricks:
        for x, z in b.footprint:
            covered[x, b.y, z] += 1
    coloured = colours != -1
    assert np.all(covered[coloured] == 1)
    assert (covered <= 1).all()

    g = build_graph(bricks)
    grounded, floating = grounded_components(g, bricks)
    assert floating == []
    assert grounded == set(range(len(bricks)))


class TestLegolizeSolidCube:
    def test_4x4x4_cube_has_no_floating_parts_and_few_bricks(self) -> None:
        colours, solid_occ = _solid_cube(4)
        bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=0, restarts=4)

        _assert_invariants(bricks, colours)
        assert len(bricks) < 20

    def test_stability_true_still_satisfies_invariants(self) -> None:
        colours, solid_occ = _solid_cube(4)
        bricks = legolize(
            colours, solid_occ, _CATALOGUE, "brick", seed=0, restarts=2, stability=True
        )

        _assert_invariants(bricks, colours)

        from image2lego.legolize.stability import analyse_stability

        assert analyse_stability(bricks, "brick").feasible


class TestLegolizeVerticalWall:
    def test_1_stud_wide_wall_has_no_seam_spanning_three_layers(self) -> None:
        # A 1-stud-wide, 4-stud-wide, 8-layer-tall wall: (x=1, y=8, z=4).
        colours = np.full((1, 8, 4), 4, dtype=np.int16)
        solid_occ = np.ones((1, 8, 4), dtype=bool)

        bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=0, restarts=2)

        _assert_invariants(bricks, colours)

        from image2lego.legolize.graph import weak_points

        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert all(len(seam) < 3 for seam in wp["seams"])


class TestLegolizeDetachedBlob:
    def test_detached_blob_is_grounded_or_dropped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A solid 4x4x3 base (thick enough to brace itself into one
        # connected structure -- a single 1-layer-thick slab wider than 2
        # studs can never self-interlock, since nothing spans its internal
        # seam), plus a detached 1x1x1 blob well above it with a solid (but
        # uncoloured) column underneath so it *can* be grounded.
        colours = np.full((4, 6, 4), -1, dtype=np.int16)
        colours[:, 0:3, :] = 4
        colours[1, 5, 1] = 6
        solid_occ = colours != -1
        solid_occ[1, 3, 1] = True
        solid_occ[1, 4, 1] = True

        with caplog.at_level(logging.WARNING):
            bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=0, restarts=2)

        # Either the blob got grounded (and every coloured voxel is still
        # covered), or it was dropped (and colours no longer requires it --
        # but since colours is fixed input, dropping would violate the
        # coverage invariant, so legolize must ground it here).
        _assert_invariants(bricks, colours)


class TestLegolizeHollowSphere:
    def test_hollowed_sphere_is_single_grounded_component(self) -> None:
        # Regression test: a hollowed sphere's shell is one connected
        # component at the *voxel* level (see
        # TestVoxelizeSphereHollowing.test_hollowed_sphere_is_one_connected_component
        # in test_voxelize.py), but greedy's per-layer largest-rectangle
        # partitioning only connects bricks via vertical footprint
        # overlap between adjacent layers -- it has no notion of the
        # underlying voxel connectivity -- so a local bulge in the shell
        # (any layer whose cross-section pokes out past the layer below
        # it) reliably gets partitioned into its own sliver with no
        # footprint overlap below, stranding it as a disconnected floating
        # component regardless of greedy's random scan corner. legolize()
        # must still produce a single grounded structure.
        sphere = trimesh.creation.icosphere(radius=150, subdivisions=3)
        occ, _ = voxelize(sphere, hollow_thickness=3, keep_components="largest")
        solid_occ, _ = voxelize(sphere, hollow_thickness=None, keep_components="largest")
        colours = np.where(occ, 4, -1).astype(np.int16)

        for seed in range(3):
            bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=seed, restarts=2)
            _assert_invariants(bricks, colours)

    def test_larger_hollow_sphere_with_adjacent_layer_disconnection(self) -> None:
        # Regression test for a second, independent bug in disconnection
        # repair: repair_weak_points seeds a bridging brick placement at
        # the disconnected component's own layer, in a window sized
        # around the midpoint of the two closest cross-component points'
        # y coordinates. When those two points are on adjacent layers
        # (common for a sphere: the closest main-component cell to a
        # floating fragment is often one layer above or below it, not on
        # the same layer), integer-averaging two adjacent y values rounds
        # toward the lower one, and the resulting window can exclude the
        # disconnected point's own layer entirely -- silently disabling
        # seeding and falling back to the near-zero-success unseeded
        # scan. At radius=180 (chosen empirically: it reproducibly
        # triggers this exact case, unlike the radius=150 sphere above,
        # which happens to trigger the *same*-layer case instead), this
        # used to fail to converge even after thousands of iterations.
        sphere = trimesh.creation.icosphere(radius=180, subdivisions=3)
        occ, _ = voxelize(sphere, hollow_thickness=3, keep_components="largest")
        solid_occ, _ = voxelize(sphere, hollow_thickness=None, keep_components="largest")
        colours = np.where(occ, 4, -1).astype(np.int16)

        for seed in range(3):
            bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=seed, restarts=2)
            _assert_invariants(bricks, colours)


class TestGreedyAllPropertyBased:
    """Coverage and non-overlap are checked directly against greedy_all
    (the piece of legolize responsible for them) across many random blobs,
    rather than through the full legolize() pipeline: legolize() also
    enforces a single-grounded-component invariant, which is a property of
    the *input geometry* (e.g. a 1-layer-thick slab wider than 2 studs can
    never self-interlock -- nothing spans its internal seam) rather than
    something arbitrary random colour data can be expected to satisfy."""

    def test_random_blobs_satisfy_coverage_and_non_overlap(self) -> None:
        rng = np.random.default_rng(123)
        for trial in range(30):
            size_x = int(rng.integers(1, 10))
            size_y = int(rng.integers(1, 6))
            size_z = int(rng.integers(1, 10))

            colours = np.full((size_x, size_y, size_z), -1, dtype=np.int16)
            mask = rng.random((size_x, size_y, size_z)) < rng.uniform(0.1, 0.9)
            colours[mask] = rng.integers(0, 4, size=mask.sum())

            bricks = greedy_all(colours, _CATALOGUE, "brick", seed=trial)

            covered = np.zeros(colours.shape, dtype=np.int32)
            for b in bricks:
                for x, z in b.footprint:
                    covered[x, b.y, z] += 1
            np.testing.assert_array_equal(covered > 0, colours != -1)
            assert (covered <= 1).all()


class TestLegolizeEmptyColours:
    def test_completely_empty_colours_returns_no_bricks(self) -> None:
        # No PALETTE / no colour data at all is a real, common state (e.g.
        # before colors.csv is supplied) -- must not crash.
        colours = np.full((4, 3, 4), -1, dtype=np.int16)
        solid_occ = np.ones((4, 3, 4), dtype=bool)

        bricks = legolize(colours, solid_occ, _CATALOGUE, "brick", seed=0, restarts=2)

        assert bricks == []
