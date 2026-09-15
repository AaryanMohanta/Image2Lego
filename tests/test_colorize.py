import numpy as np
import pytest
import trimesh

from image2lego.colours import Colour
from image2lego.geometry.colorize import colorize, sample_mesh_colours, surface_mask

_RED = Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(255, 0, 0))
_WHITE = Colour(ldraw_id=15, bricklink_id=1, name="White", rgb=(255, 255, 255))
_BLUE = Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0, 0, 255))
_BLACK = Colour(ldraw_id=0, bricklink_id=11, name="Black", rgb=(0, 0, 0))


class TestSurfaceMask:
    def test_solid_cube_hollow_shell_only(self) -> None:
        occ = np.ones((3, 3, 3), dtype=bool)
        mask = surface_mask(occ)
        assert mask[1, 1, 1] == np.False_  # centre voxel: all 6 neighbours occupied
        assert mask.sum() == 26  # every voxel except the centre

    def test_empty_grid_has_no_surface(self) -> None:
        occ = np.zeros((3, 3, 3), dtype=bool)
        assert not surface_mask(occ).any()

    def test_single_voxel_is_surface(self) -> None:
        occ = np.ones((1, 1, 1), dtype=bool)
        assert surface_mask(occ).all()

    def test_never_marks_empty_voxels_as_surface(self) -> None:
        occ = np.zeros((3, 3, 3), dtype=bool)
        occ[1, 1, 1] = True
        mask = surface_mask(occ)
        assert mask[1, 1, 1]
        assert mask.sum() == 1


def _solid_texture_box(
    extents: tuple[float, float, float], colour: tuple[int, int, int]
) -> trimesh.Trimesh:
    from PIL import Image

    box = trimesh.creation.box(extents=extents)
    image = Image.new("RGB", (4, 4), color=colour)
    uv = np.zeros((len(box.vertices), 2))
    material = trimesh.visual.material.SimpleMaterial(image=image)
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material, image=image)
    return box


def _two_tone_vertex_box(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    colours = np.zeros((len(box.vertices), 4), dtype=np.uint8)
    colours[:, 3] = 255
    left = box.vertices[:, 0] < 0
    colours[left] = [*_BLUE.rgb, 255]
    colours[~left] = [*_WHITE.rgb, 255]
    box.visual.vertex_colors = colours
    return box


class TestSampleMeshColours:
    def test_samples_solid_texture_colour(self) -> None:
        mesh = _solid_texture_box((40, 40, 40), (255, 0, 0))
        points = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 5]])
        rgb = sample_mesh_colours(mesh, points, np.eye(4))
        assert rgb.shape == (3, 3)
        assert rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb, np.tile([255, 0, 0], (3, 1)))

    def test_falls_back_to_mid_grey_with_no_colour_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _NoColourVisual:
            mesh = None

        mesh = trimesh.creation.box(extents=(40, 40, 40))
        mesh.visual = _NoColourVisual()
        points = np.array([[0, 0, 0]])
        with caplog.at_level("WARNING"):
            rgb = sample_mesh_colours(mesh, points, np.eye(4))
        np.testing.assert_array_equal(rgb, np.array([[128, 128, 128]]))
        assert "colour" in caplog.text.lower()

    def test_applies_inverse_transform_before_querying(self) -> None:
        mesh = _solid_texture_box((40, 40, 40), (0, 255, 0))
        # transform doubles the mesh into voxel space; a voxel-space point of
        # (10, 0, 0) should map back to (5, 0, 0) in the mesh's own space,
        # which is still well inside this 40-wide box.
        transform = np.diag([2.0, 2.0, 2.0, 1.0])
        points = np.array([[10.0, 0.0, 0.0]])
        rgb = sample_mesh_colours(mesh, points, transform)
        np.testing.assert_array_equal(rgb, np.array([[0, 255, 0]]))


class TestColorizeUnitCube:
    def test_solid_red_texture_colorizes_to_red(self) -> None:
        mesh = _solid_texture_box((20, 20, 20), (255, 0, 0))
        occ = np.ones((1, 1, 1), dtype=bool)
        palette = [_RED, _WHITE, _BLUE, _BLACK]

        grid = colorize(occ, mesh, np.eye(4), palette)

        assert grid.shape == (1, 1, 1)
        assert grid[0, 0, 0] == _RED.ldraw_id


class TestColorizeTwoTone:
    def test_left_half_blue_right_half_white_with_tolerance(self) -> None:
        mesh = _two_tone_vertex_box((80, 80, 80))
        occ = np.ones((4, 4, 4), dtype=bool)
        palette = [_RED, _WHITE, _BLUE, _BLACK]

        grid = colorize(occ, mesh, np.eye(4), palette)

        left = grid[0:2, :, :]
        right = grid[2:4, :, :]

        left_match = (left == _BLUE.ldraw_id).mean()
        right_match = (right == _WHITE.ldraw_id).mean()

        assert left_match >= 0.8
        assert right_match >= 0.8


class TestColorizeMaxColours:
    def test_max_colours_one_yields_single_colour(self) -> None:
        mesh = _two_tone_vertex_box((80, 80, 80))
        occ = np.ones((4, 4, 4), dtype=bool)
        palette = [_RED, _WHITE, _BLUE, _BLACK]

        grid = colorize(occ, mesh, np.eye(4), palette, max_colours=1)

        used = np.unique(grid[grid != -1])
        assert len(used) == 1
