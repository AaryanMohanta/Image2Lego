"""Sample a mesh's surface colour (texture or vertex colours) onto a voxel
occupancy grid, and snap it to a fixed brick-colour palette.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import scipy.ndimage
import trimesh

from image2lego.colours import Colour, nearest_colour_batch, srgb_array_to_lab
from image2lego.geometry.voxelize import voxel_centres
from image2lego.model import STUD_LDU, ColourGrid, Occupancy

logger = logging.getLogger(__name__)

InteriorFill = Literal["nearest", "filler"]

_STRUCTURE_6_CONNECTED = scipy.ndimage.generate_binary_structure(3, 1)

# Sub-voxel offsets (as a fraction of the stud pitch) used to take 4 jittered
# colour samples per surface voxel instead of just its centre.
_JITTER_FRACTIONS = np.array(
    [
        [0.25, 0.25, 0.25],
        [0.25, -0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25],
    ]
)


def surface_mask(occ: Occupancy) -> Occupancy:
    """True for occupied voxels that have at least one empty (or
    out-of-grid) 6-connected neighbour."""
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    exposed = np.zeros_like(occ, dtype=bool)
    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        neighbour = padded[
            1 + dx : 1 + dx + occ.shape[0],
            1 + dy : 1 + dy + occ.shape[1],
            1 + dz : 1 + dz + occ.shape[2],
        ]
        exposed |= ~neighbour
    return occ & exposed


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    return (homogeneous @ matrix.T)[..., :3]


def _barycentric(mesh: trimesh.Trimesh, points: np.ndarray, face_index: np.ndarray) -> np.ndarray:
    triangles = mesh.triangles[face_index]
    return trimesh.triangles.points_to_barycentric(triangles, points)


def _sample_texture_colours(
    mesh: trimesh.Trimesh,
    visual: trimesh.visual.texture.TextureVisuals,
    points: np.ndarray,
    face_index: np.ndarray,
) -> np.ndarray:
    bary = _barycentric(mesh, points, face_index)
    uv = visual.uv
    face_uv = uv[mesh.faces[face_index]]
    interpolated_uv = np.einsum("ij,ijk->ik", bary, face_uv)

    image = visual.material.image
    image_array = np.asarray(image.convert("RGB"))
    height, width = image_array.shape[:2]

    px = np.clip((interpolated_uv[:, 0] * width).astype(int), 0, width - 1)
    py = np.clip(((1 - interpolated_uv[:, 1]) * height).astype(int), 0, height - 1)
    return image_array[py, px].astype(np.uint8)


def _sample_vertex_colours(
    mesh: trimesh.Trimesh,
    visual: trimesh.visual.color.ColorVisuals,
    points: np.ndarray,
    face_index: np.ndarray,
) -> np.ndarray:
    bary = _barycentric(mesh, points, face_index)
    vertex_colours = visual.vertex_colors[:, :3].astype(np.float64)
    face_colours = vertex_colours[mesh.faces[face_index]]
    interpolated = np.einsum("ij,ijk->ik", bary, face_colours)
    return np.clip(interpolated, 0, 255).astype(np.uint8)


def sample_mesh_colours(
    mesh: trimesh.Trimesh, points_ldu: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Sample mesh surface colour at query points given in voxel/LDU space.

    mesh and transform should be the pair from scale_mesh_to_studs's
    caller *before* that scaling was applied -- i.e. mesh is the
    oriented-but-unscaled mesh, and transform is the scale matrix
    scale_mesh_to_studs returned. points_ldu are mapped through transform's
    inverse before querying, so nearest-surface distances are computed in
    the mesh's own (isometric, undistorted) coordinate space rather than
    the possibly-anisotropically-scaled voxel space.

    Returns (N, 3) uint8 RGB.
    """
    points_original = _transform_points(points_ldu, np.linalg.inv(transform))
    closest, _distance, face_index = mesh.nearest.on_surface(points_original)

    visual = mesh.visual
    if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.uv is not None:
        return _sample_texture_colours(mesh, visual, closest, face_index)
    if isinstance(visual, trimesh.visual.color.ColorVisuals):
        return _sample_vertex_colours(mesh, visual, closest, face_index)

    logger.warning("mesh has no usable colour information; using mid grey")
    return np.full((len(points_original), 3), 128, dtype=np.uint8)


def _kmeans_replace_with_cluster_mean(rgb: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cluster (N, 3) uint8 RGB colours into up to k groups by Lab distance,
    replacing every point by the mean RGB of its cluster. Tiny numpy-only
    k-means; no sklearn."""
    n = len(rgb)
    k = max(1, min(k, n))
    if k == 1:
        mean = rgb.mean(axis=0)
        return np.tile(mean, (n, 1)).astype(np.uint8)

    lab = srgb_array_to_lab(rgb)
    rng = np.random.default_rng(seed)
    centre_indices = rng.choice(n, size=k, replace=False)
    centres = lab[centre_indices]

    assignments = np.zeros(n, dtype=int)
    for _ in range(20):
        distances = np.sum((lab[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(new_assignments, assignments) and _ > 0:
            break
        assignments = new_assignments
        for i in range(k):
            mask = assignments == i
            if np.any(mask):
                centres[i] = lab[mask].mean(axis=0)

    result = np.zeros_like(rgb)
    for i in range(k):
        mask = assignments == i
        if np.any(mask):
            result[mask] = rgb[mask].mean(axis=0).astype(np.uint8)
    return result


def _majority_vote_pass(
    colour_grid: ColourGrid, surface: Occupancy, colour_ids: list[int]
) -> ColourGrid:
    shape = colour_grid.shape
    padded_colours = np.pad(colour_grid, 1, mode="constant", constant_values=-1)
    padded_surface = np.pad(surface, 1, mode="constant", constant_values=False)

    stack = [colour_grid]
    for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        neighbour = padded_colours[
            1 + dx : 1 + dx + shape[0], 1 + dy : 1 + dy + shape[1], 1 + dz : 1 + dz + shape[2]
        ]
        neighbour_surface = padded_surface[
            1 + dx : 1 + dx + shape[0], 1 + dy : 1 + dy + shape[1], 1 + dz : 1 + dz + shape[2]
        ]
        stack.append(np.where(neighbour_surface, neighbour, -1))
    stacked = np.stack(stack, axis=0)

    best_value = np.full(shape, -1, dtype=colour_grid.dtype)
    best_count = np.zeros(shape, dtype=np.int32)
    for colour_id in colour_ids:
        count = np.sum(stacked == colour_id, axis=0)
        take = count > best_count
        best_value = np.where(take, colour_id, best_value)
        best_count = np.where(take, count, best_count)

    new_grid = colour_grid.copy()
    new_grid[surface] = best_value[surface]
    return new_grid


def colorize(
    occ: Occupancy,
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    palette: list[Colour],
    max_colours: int | None = None,
    interior: InteriorFill = "nearest",
    filler_colour: int = 71,
    smooth_passes: int = 1,
) -> ColourGrid:
    """Colour every occupied voxel: sample the mesh surface colour under
    each surface voxel, snap to the nearest palette colour, smooth, then
    fill interior voxels from their nearest coloured surface voxel (or a
    flat filler colour).

    mesh/transform are the same pair sample_mesh_colours expects: mesh is
    the oriented-but-unscaled mesh scale_mesh_to_studs was called on, and
    transform is the scale matrix it returned. This lets colorize
    reconstruct the scaled mesh's bounding box (mesh.bounds transformed by
    `transform`) to align voxel centres with occ, without needing occ's own
    grid offset -- assuming occ was built the standard way (occ[:, 0, :]
    already touching the model's actual lowest layer; see voxelize.py's
    docstring for the one-layer-offset caveat this shares).
    """
    colour_grid: ColourGrid = np.full(occ.shape, -1, dtype=np.int16)
    surface = surface_mask(occ)
    if not surface.any():
        return colour_grid

    mins = _transform_points(mesh.bounds[0][None, :], transform)[0]
    centres = voxel_centres(occ.shape, mins)
    surface_centres = centres[surface]

    jitter = _JITTER_FRACTIONS * STUD_LDU
    jittered_points = (surface_centres[:, None, :] + jitter[None, :, :]).reshape(-1, 3)
    jittered_rgb = sample_mesh_colours(mesh, jittered_points, transform)
    samples = jittered_rgb.reshape(len(surface_centres), len(jitter), 3).astype(np.float64)
    sample_rgb = samples.mean(axis=1).astype(np.uint8)

    if max_colours is not None:
        sample_rgb = _kmeans_replace_with_cluster_mean(sample_rgb, max_colours)

    sample_ids = nearest_colour_batch(sample_rgb, palette)
    colour_grid[surface] = sample_ids.astype(np.int16)

    colour_ids = sorted({c.ldraw_id for c in palette})
    for _ in range(smooth_passes):
        colour_grid = _majority_vote_pass(colour_grid, surface, colour_ids)

    interior_mask = occ & ~surface
    if interior_mask.any():
        if interior == "filler":
            colour_grid = np.where(interior_mask, np.int16(filler_colour), colour_grid).astype(
                np.int16
            )
        else:
            _distance, indices = scipy.ndimage.distance_transform_edt(
                ~surface, return_indices=True
            )
            nearest_surface_colour = colour_grid[indices[0], indices[1], indices[2]]
            colour_grid = np.where(interior_mask, nearest_surface_colour, colour_grid).astype(
                np.int16
            )

    return colour_grid
