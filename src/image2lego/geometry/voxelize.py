"""Turn a mesh (already scaled to LDU/stud space by scale_mesh_to_studs)
into a dense boolean occupancy grid.

Axis order: occupancy is indexed [x, y, z], matching the model convention
where x/z are the horizontal stud grid and y is the layer index counting
up from 0 at ground level (occ[:, 0, :] is the bottom layer).

trimesh's own voxelized().fill() pads its grid generously and isn't
reliable for getting an exact stud-aligned shape, so instead we resample
its filled dense grid onto our own grid, one cell per stud/layer, sized
directly from the (already stud-scaled) mesh's bounding box.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import scipy.ndimage
import trimesh

from image2lego.model import STUD_LDU, Occupancy

logger = logging.getLogger(__name__)

KeepComponents = Literal["largest", "all"]

_STRUCTURE_6_CONNECTED = scipy.ndimage.generate_binary_structure(3, 1)


def grid_shape_for_bounds(extents: np.ndarray) -> np.ndarray:
    """The [x, y, z] voxel counts a mesh with these extents occupies at
    pitch=STUD_LDU (at least 1 cell per axis)."""
    return np.maximum(np.round(extents / STUD_LDU).astype(int), 1)


def voxel_centres(shape: tuple[int, int, int] | np.ndarray, mins: np.ndarray) -> np.ndarray:
    """World-space (LDU) centres of every cell in a [x, y, z] grid of this
    shape, whose index (0, 0, 0) cell's minimum corner sits at `mins`.
    Returns an (X, Y, Z, 3) array."""
    centres = [mins[axis] + (np.arange(shape[axis]) + 0.5) * STUD_LDU for axis in range(3)]
    gx, gy, gz = np.meshgrid(*centres, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)


def _resample_to_stud_grid(mesh: trimesh.Trimesh) -> tuple[Occupancy, np.ndarray]:
    filled = mesh.voxelized(pitch=STUD_LDU).fill()
    source = filled.matrix

    mins = mesh.bounds[0]
    shape = grid_shape_for_bounds(mesh.extents)

    centres = voxel_centres(shape, mins)
    world = np.concatenate([centres, np.ones((*centres.shape[:3], 1))], axis=-1)

    inverse_transform = np.linalg.inv(filled.transform)
    index = world @ inverse_transform.T
    ix = np.round(index[..., 0]).astype(int)
    iy = np.round(index[..., 1]).astype(int)
    iz = np.round(index[..., 2]).astype(int)

    valid = (
        (ix >= 0)
        & (ix < source.shape[0])
        & (iy >= 0)
        & (iy < source.shape[1])
        & (iz >= 0)
        & (iz < source.shape[2])
    )

    occ = np.zeros(tuple(shape), dtype=bool)
    occ[valid] = source[ix[valid], iy[valid], iz[valid]]

    offset = mins / STUD_LDU
    return occ, offset


def _hollow(occ: Occupancy, thickness: int) -> Occupancy:
    eroded = scipy.ndimage.binary_erosion(
        occ, structure=_STRUCTURE_6_CONNECTED, iterations=thickness
    )
    return occ & ~eroded


def _keep_largest_component(occ: Occupancy) -> Occupancy:
    labeled, num_features = scipy.ndimage.label(occ, structure=_STRUCTURE_6_CONNECTED)
    if num_features <= 1:
        return occ

    sizes = scipy.ndimage.sum(occ, labeled, index=range(1, num_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    kept = labeled == largest_label

    dropped = int(occ.sum() - kept.sum())
    logger.info(
        "dropped %d voxels from %d smaller connected component(s)",
        dropped,
        num_features - 1,
    )
    return kept


def _ensure_touches_ground(occ: Occupancy, offset: np.ndarray) -> tuple[Occupancy, np.ndarray]:
    if occ.shape[1] == 0 or not occ.any():
        return occ, offset
    if occ[:, 0, :].any():
        return occ, offset

    occupied_layers = np.nonzero(occ.any(axis=(0, 2)))[0]
    lowest = int(occupied_layers[0])
    shifted = occ[:, lowest:, :]
    new_offset = offset.copy()
    new_offset[1] += lowest
    return shifted, new_offset


def voxelize(
    mesh: trimesh.Trimesh,
    hollow_thickness: int | None = 3,
    keep_components: KeepComponents = "largest",
) -> tuple[Occupancy, np.ndarray]:
    """Voxelize mesh at pitch=STUD_LDU into a dense [x, y, z] occupancy grid.

    Returns (occupancy, offset), where offset is the (x, y, z) stud/layer
    coordinate of the mesh's bounding-box minimum corner -- i.e. the world
    position (in stud/layer units) that occupancy index (0, 0, 0) sits at.
    """
    occ, offset = _resample_to_stud_grid(mesh)

    if hollow_thickness is not None:
        occ = _hollow(occ, hollow_thickness)

    if keep_components == "largest":
        occ = _keep_largest_component(occ)

    occ, offset = _ensure_touches_ground(occ, offset)

    return occ, offset


def summarise(occ: Occupancy) -> dict[str, object]:
    """Return dims, voxel count, layer count and a rough estimated brick
    count (voxels/3) for the CLI to report / warn on."""
    voxel_count = int(occ.sum())
    if voxel_count:
        layer_count = int(np.nonzero(occ.any(axis=(0, 2)))[0].max()) + 1
    else:
        layer_count = 0

    return {
        "dims": tuple(int(n) for n in occ.shape),
        "voxel_count": voxel_count,
        "layer_count": layer_count,
        "estimated_brick_count": -(-voxel_count // 3),
    }
