from pathlib import Path

import numpy as np
import pytest
import trimesh

from image2lego.geometry.mesh import load_mesh, orient_mesh, scale_mesh_to_studs
from image2lego.model import BRICK_H_LDU, PLATE_H_LDU, STUD_LDU


class TestLoadMesh:
    def test_loads_stl_and_repairs(self, tmp_path: Path) -> None:
        box = trimesh.creation.box(extents=(80, 40, 120))
        path = tmp_path / "box.stl"
        box.export(path)

        mesh = load_mesh(path)

        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(80 * 40 * 120, rel=1e-3)

    def test_concatenates_scene_into_single_mesh(self, tmp_path: Path) -> None:
        box_a = trimesh.creation.box(extents=(40, 40, 40))
        box_b = trimesh.creation.box(extents=(40, 40, 40))
        box_b.apply_translation([100, 0, 0])
        scene = trimesh.Scene([box_a, box_b])
        path = tmp_path / "scene.glb"
        scene.export(path)

        mesh = load_mesh(path)

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0


class TestOrientMesh:
    def test_auto_picks_smallest_extent_axis_as_up(self) -> None:
        # x is the smallest extent (10), so orient_mesh should rotate x onto +Y.
        box = trimesh.creation.box(extents=(10, 100, 200))
        oriented = orient_mesh(box, up="auto")
        assert oriented.extents[1] == pytest.approx(10, abs=1e-6)

    def test_auto_leaves_already_y_up_mesh_alone(self) -> None:
        box = trimesh.creation.box(extents=(100, 10, 200))
        oriented = orient_mesh(box, up="auto")
        assert oriented.extents[1] == pytest.approx(10, abs=1e-6)

    def test_centers_xz_and_grounds_y(self) -> None:
        box = trimesh.creation.box(extents=(10, 100, 200))
        oriented = orient_mesh(box, up="auto")
        bounds = oriented.bounds
        assert bounds[0][1] == pytest.approx(0.0, abs=1e-6)
        center_x = (bounds[0][0] + bounds[1][0]) / 2
        center_z = (bounds[0][2] + bounds[1][2]) / 2
        assert center_x == pytest.approx(0.0, abs=1e-6)
        assert center_z == pytest.approx(0.0, abs=1e-6)

    def test_explicit_up_overrides_auto_heuristic(self) -> None:
        # Smallest extent is x (10), but we force z to become up instead.
        box = trimesh.creation.box(extents=(10, 100, 200))
        oriented = orient_mesh(box, up="z")
        assert oriented.extents[1] == pytest.approx(200, abs=1e-6)

    def test_explicit_up_y_is_a_no_op_rotation(self) -> None:
        box = trimesh.creation.box(extents=(10, 100, 200))
        oriented = orient_mesh(box, up="y")
        assert oriented.extents[1] == pytest.approx(100, abs=1e-6)


class TestScaleMeshToStuds:
    def _make_oriented_box(self) -> trimesh.Trimesh:
        # Already y-up, centred at origin with min_y = 0: x=40, y=30, z=60.
        box = trimesh.creation.box(extents=(40, 30, 60))
        box.apply_translation([0, 15, 0])
        return box

    def test_scales_longer_xz_extent_to_width_studs(self) -> None:
        mesh = self._make_oriented_box()
        scaled, _transform = scale_mesh_to_studs(mesh, width_studs=6, layer_type="brick")
        assert max(scaled.extents[0], scaled.extents[2]) == pytest.approx(
            6 * STUD_LDU, rel=1e-6
        )

    def test_xz_scale_is_uniform(self) -> None:
        mesh = self._make_oriented_box()
        scaled, _transform = scale_mesh_to_studs(mesh, width_studs=6, layer_type="brick")
        xz_scale = (6 * STUD_LDU) / 60
        assert scaled.extents[0] == pytest.approx(40 * xz_scale, rel=1e-6)
        assert scaled.extents[2] == pytest.approx(60 * xz_scale, rel=1e-6)

    def test_y_extent_scaled_by_stud_over_brick_height(self) -> None:
        mesh = self._make_oriented_box()
        scaled, transform = scale_mesh_to_studs(mesh, width_studs=6, layer_type="brick")
        xz_scale = (6 * STUD_LDU) / 60
        expected_y_scale = xz_scale * (STUD_LDU / BRICK_H_LDU)
        assert scaled.extents[1] == pytest.approx(30 * expected_y_scale, rel=1e-6)
        assert transform[1, 1] == pytest.approx(expected_y_scale, rel=1e-6)

    def test_y_extent_scaled_by_stud_over_plate_height(self) -> None:
        mesh = self._make_oriented_box()
        scaled, transform = scale_mesh_to_studs(mesh, width_studs=6, layer_type="plate")
        xz_scale = (6 * STUD_LDU) / 60
        expected_y_scale = xz_scale * (STUD_LDU / PLATE_H_LDU)
        assert scaled.extents[1] == pytest.approx(30 * expected_y_scale, rel=1e-6)
        assert transform[1, 1] == pytest.approx(expected_y_scale, rel=1e-6)

    def test_transform_is_diagonal_scale_matrix(self) -> None:
        mesh = self._make_oriented_box()
        _scaled, transform = scale_mesh_to_studs(mesh, width_studs=6, layer_type="brick")
        assert transform.shape == (4, 4)
        off_diagonal = transform.copy()
        np.fill_diagonal(off_diagonal, 0)
        assert np.allclose(off_diagonal, 0)
        assert transform[3, 3] == pytest.approx(1.0)
