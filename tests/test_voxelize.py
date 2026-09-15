import numpy as np
import scipy.ndimage
import trimesh

from image2lego.geometry.voxelize import summarise, voxelize


class TestVoxelizeBox:
    def test_4x2x6_stud_box_has_exact_shape_and_count(self) -> None:
        # A box already sized as if scale_mesh_to_studs had targeted
        # width_studs=6 (longer xz extent = 6 studs = 120 LDU) with a y
        # extent of 2 pitch-units (2 layers).
        box = trimesh.creation.box(extents=(4 * 20, 2 * 20, 6 * 20))

        occ, offset = voxelize(box, hollow_thickness=None, keep_components="largest")

        assert occ.shape == (4, 2, 6)
        assert occ.sum() == 4 * 2 * 6
        assert offset.shape == (3,)

    def test_touches_layer_zero(self) -> None:
        box = trimesh.creation.box(extents=(4 * 20, 2 * 20, 6 * 20))
        occ, _offset = voxelize(box, hollow_thickness=None, keep_components="largest")
        assert occ[:, 0, :].any()


class TestVoxelizeSphereHollowing:
    def _sphere(self) -> trimesh.Trimesh:
        # Radius chosen so the sphere spans ~15 voxels across (diameter
        # 300 LDU / pitch 20), enough room for a hollow_thickness=3 shell
        # to survive without collapsing.
        return trimesh.creation.icosphere(radius=150, subdivisions=3)

    def test_hollowing_reduces_voxel_count(self) -> None:
        sphere = self._sphere()
        solid, _ = voxelize(sphere, hollow_thickness=None, keep_components="largest")
        hollow, _ = voxelize(sphere, hollow_thickness=3, keep_components="largest")

        assert hollow.sum() > 0
        assert hollow.sum() < solid.sum()

    def test_hollowed_sphere_is_one_connected_component(self) -> None:
        sphere = self._sphere()
        hollow, _ = voxelize(sphere, hollow_thickness=3, keep_components="largest")

        structure = scipy.ndimage.generate_binary_structure(3, 1)
        _labeled, num_features = scipy.ndimage.label(hollow, structure=structure)
        assert num_features == 1


class TestSummarise:
    def test_reports_dims_count_layers_and_estimated_bricks(self) -> None:
        occ = np.zeros((4, 3, 6), dtype=bool)
        occ[:, 0, :] = True
        occ[:2, 1, :] = True

        summary = summarise(occ)

        assert summary["dims"] == (4, 3, 6)
        assert summary["voxel_count"] == 24 + 12
        assert summary["layer_count"] == 2
        assert summary["estimated_brick_count"] == -(-(24 + 12) // 3)

    def test_empty_occupancy_has_zero_layers(self) -> None:
        occ = np.zeros((2, 2, 2), dtype=bool)
        summary = summarise(occ)
        assert summary["voxel_count"] == 0
        assert summary["layer_count"] == 0
