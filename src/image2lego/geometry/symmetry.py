"""Force a voxel grid to be left-right symmetric across a horizontal axis."""

from __future__ import annotations

from typing import Literal

import numpy as np

from image2lego.model import Occupancy

SymmetryAxis = Literal["x", "z"]

_AXIS_INDEX: dict[SymmetryAxis, int] = {"x": 0, "z": 2}


def symmetrise(occ: Occupancy, axis: SymmetryAxis = "x") -> Occupancy:
    """Mirror-symmetrise occ across the given horizontal axis: compare the
    total voxel count of each half, keep the half with more voxels, and
    overwrite the other half with its mirror (an odd-sized axis's unpaired
    centre slice is left as-is)."""
    axis_index = _AXIS_INDEX[axis]
    moved = np.moveaxis(occ, axis_index, 0).copy()
    size = moved.shape[0]
    half = size // 2

    left = moved[:half]
    right = moved[size - half :]

    if left.sum() >= right.sum():
        moved[size - half :] = left[::-1]
    else:
        moved[:half] = right[::-1]

    return np.moveaxis(moved, 0, axis_index)
