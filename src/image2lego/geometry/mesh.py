"""Load, repair, orient and scale a source mesh into LDU (LEGO stud) space."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from image2lego.model import BRICK_H_LDU, PLATE_H_LDU, STUD_LDU, LayerType

logger = logging.getLogger(__name__)

UpAxis = Literal["auto", "x", "y", "z"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a glb/gltf/obj/stl/ply file into a single repaired Trimesh."""
    loaded = trimesh.load(path)

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path}: expected a mesh, got {type(mesh).__name__}")

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())

    if not mesh.is_watertight:
        mesh.fill_holes()

    if mesh.is_watertight:
        logger.info("mesh %s is watertight after repair", path)
    else:
        logger.warning("mesh %s is NOT watertight after repair", path)

    return mesh


def _rotation_to_y_up(axis: Literal["x", "y", "z"]) -> np.ndarray:
    if axis == "y":
        return np.eye(4)
    if axis == "x":
        return trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    return trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def orient_mesh(mesh: trimesh.Trimesh, up: UpAxis = "auto") -> trimesh.Trimesh:
    """Rotate mesh so +Y is up, then centre it on the origin in x/z and put
    its lowest point at y=0.

    "auto" picks the axis with the smallest extent as the current up axis
    (the thin dimension of an object lying on its side is usually the one
    that should end up vertical).
    """
    if up == "auto":
        axis_names: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
        axis_index = int(np.argmin(mesh.extents))
        axis = axis_names[axis_index]
    else:
        axis = up

    mesh.apply_transform(_rotation_to_y_up(axis))

    bounds = mesh.bounds
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    center_z = (bounds[0][2] + bounds[1][2]) / 2
    min_y = bounds[0][1]
    mesh.apply_translation([-center_x, -min_y, -center_z])

    return mesh


def scale_mesh_to_studs(
    mesh: trimesh.Trimesh, width_studs: int, layer_type: LayerType
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Scale mesh (already y-up, origin-centred) so the longer of its x/z
    extents equals width_studs studs, then additionally scale y so that a
    single voxel of pitch STUD_LDU represents one brick/plate layer.

    Returns the scaled mesh and the 4x4 transform that was applied, so
    later steps can map a voxel back to the original mesh's surface for
    texture-colour sampling.
    """
    h = BRICK_H_LDU if layer_type == "brick" else PLATE_H_LDU
    xz_extent = max(mesh.extents[0], mesh.extents[2])
    xz_scale = (width_studs * STUD_LDU) / xz_extent
    y_scale = xz_scale * (STUD_LDU / h)

    transform = np.diag([xz_scale, y_scale, xz_scale, 1.0])
    mesh.apply_transform(transform)

    return mesh, transform
