"""Render a brick model as a shaded 3/4-view PNG.

Uses a hand-rolled isometric-style painter's-algorithm projection rather
than matplotlib's mplot3d Axes3D.voxels(): voxels() builds a separate 3D
patch (with its own hidden-surface handling) per unit cube face, which
becomes impractically slow well before a few thousand bricks. Instead,
each brick is drawn as up to 3 visible flat-shaded quads (top + two sides),
depth-sorted back-to-front at the brick level and batched into a single
matplotlib PolyCollection, so a few thousand bricks render in well under a
second.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from image2lego.colours import Colour
from image2lego.model import BRICK_H_LDU, PLATE_H_LDU, STUD_LDU, Brick, LayerType

# Classic isometric-style projection: the camera looks from the (+,+,+)
# octant toward the origin, so +x, +y (up) and +z are all simultaneously
# toward it and depth is just their sum.
_COS30 = math.cos(math.radians(30))
_SIN30 = math.sin(math.radians(30))

# Flat shading multipliers per visible face (top brightest, then the two
# visible sides), applied to each brick's base RGB -- the classic 3-shades-
# per-cube look used in isometric voxel art.
_TOP_SHADE = 1.0
_PLUS_X_SHADE = 0.72
_PLUS_Z_SHADE = 0.5

_EDGE_COLOUR = (0.8, 0.8, 0.8)  # light grey, so adjacent same-colour bricks stay distinguishable
_BACKGROUND = "white"
_FALLBACK_RGB = (160, 160, 160)

# Corner offsets (as 0/1 flags along x, y, z) for a unit box, and which
# corners make up each of the 3 faces visible from this camera direction
# (the camera looks roughly from +x, +y, +z toward the origin, so we see
# the top face and the two faces whose outward normal has a positive x or
# z component).
_TOP_FACE = ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))
_PLUS_X_FACE = ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))
_PLUS_Z_FACE = ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def _project(points: np.ndarray) -> np.ndarray:
    """World (x, y, z) -> camera-space (screen_x, screen_y, depth), for an
    array of points shaped (..., 3). depth increases toward the camera, so
    sorting ascending gives back-to-front (painter's algorithm) order."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    screen_x = (x - z) * _COS30
    screen_y = (x + z) * _SIN30 - y
    depth = x + y + z
    return np.stack([screen_x, screen_y, depth], axis=-1)


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[float, float, float]:
    r, g, b = rgb
    return (
        min(1.0, r / 255.0 * factor),
        min(1.0, g / 255.0 * factor),
        min(1.0, b / 255.0 * factor),
    )


def render_preview(
    bricks: list[Brick],
    palette: dict[int, Colour],
    path: str | os.PathLike[str],
    layer_type: LayerType = "brick",
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 150,
) -> None:
    """Render `bricks` (coloured by `palette`, a {ldraw_id: Colour} map --
    falls back to mid-grey for an unknown id) as a shaded 3/4-view PNG at
    `path`."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(_BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")

    if not bricks:
        ax.text(0.5, 0.5, "(empty model)", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(path, facecolor=_BACKGROUND)
        plt.close(fig)
        return

    h = BRICK_H_LDU if layer_type == "brick" else PLATE_H_LDU

    face_verts: list[np.ndarray] = []
    face_colours: list[tuple[float, float, float]] = []
    face_depths: list[float] = []
    all_screen_xy: list[np.ndarray] = []

    for brick in bricks:
        x0, x1 = brick.x * STUD_LDU, (brick.x + brick.w) * STUD_LDU
        z0, z1 = brick.z * STUD_LDU, (brick.z + brick.d) * STUD_LDU
        y0, y1 = brick.y * h, (brick.y + 1) * h
        lo = np.array([x0, y0, z0])
        span = np.array([x1 - x0, y1 - y0, z1 - z0])

        colour = palette.get(brick.colour)
        rgb = colour.rgb if colour is not None else _FALLBACK_RGB

        centroid = lo + span / 2
        depth = float(_project(centroid)[2])

        visible_faces = (
            (_TOP_FACE, _TOP_SHADE),
            (_PLUS_X_FACE, _PLUS_X_SHADE),
            (_PLUS_Z_FACE, _PLUS_Z_SHADE),
        )
        for face, shade in visible_faces:
            world_pts = lo + np.array(face) * span
            screen_pts = _project(world_pts)[:, :2]
            face_verts.append(screen_pts)
            face_colours.append(_shade(rgb, shade))
            face_depths.append(depth)
            all_screen_xy.append(screen_pts)

    order = np.argsort(face_depths)
    verts_sorted = [face_verts[i] for i in order]
    colours_sorted = [face_colours[i] for i in order]

    collection = PolyCollection(
        verts_sorted,
        facecolors=colours_sorted,
        edgecolors=[_EDGE_COLOUR],
        linewidths=0.4,
    )
    ax.add_collection(collection)

    all_xy = np.concatenate(all_screen_xy, axis=0)
    x_range = all_xy[:, 0].max() - all_xy[:, 0].min()
    y_range = all_xy[:, 1].max() - all_xy[:, 1].min()
    margin = max(x_range, y_range) * 0.05 + 1
    ax.set_xlim(all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin)
    ax.set_ylim(all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin)

    fig.tight_layout(pad=0.2)
    fig.savefig(path, facecolor=_BACKGROUND)
    plt.close(fig)
