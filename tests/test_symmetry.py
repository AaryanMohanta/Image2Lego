import numpy as np

from image2lego.geometry.symmetry import symmetrise


class TestSymmetrise:
    def test_result_equals_its_own_mirror_across_x(self) -> None:
        rng = np.random.default_rng(0)
        occ = rng.random((6, 4, 5)) < 0.5

        result = symmetrise(occ, axis="x")

        np.testing.assert_array_equal(result, np.flip(result, axis=0))

    def test_result_equals_its_own_mirror_across_z(self) -> None:
        rng = np.random.default_rng(1)
        occ = rng.random((5, 3, 7)) < 0.5

        result = symmetrise(occ, axis="z")

        np.testing.assert_array_equal(result, np.flip(result, axis=2))

    def test_keeps_the_half_with_more_voxels(self) -> None:
        occ = np.zeros((4, 1, 1), dtype=bool)
        occ[0:2] = True  # left half fully occupied
        occ[2:4] = False  # right half empty -> left half has more voxels

        result = symmetrise(occ, axis="x")

        # left half (more voxels) is kept and mirrored onto the right.
        np.testing.assert_array_equal(result[:, 0, 0], [True, True, True, True])

    def test_odd_axis_length_leaves_centre_slice_untouched(self) -> None:
        occ = np.zeros((5, 1, 1), dtype=bool)
        occ[0] = True  # only the far-left cell set; left half > right half
        occ[2] = True  # centre slice

        result = symmetrise(occ, axis="x")

        assert result[2, 0, 0] == occ[2, 0, 0]

    def test_already_symmetric_grid_is_unchanged(self) -> None:
        occ = np.zeros((4, 2, 2), dtype=bool)
        occ[0] = True
        occ[3] = True  # mirror of index 0

        result = symmetrise(occ, axis="x")

        np.testing.assert_array_equal(result, occ)

    def test_preserves_shape(self) -> None:
        occ = np.ones((6, 3, 5), dtype=bool)
        result = symmetrise(occ, axis="x")
        assert result.shape == occ.shape
