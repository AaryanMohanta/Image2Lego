"""Colour catalogue built from a Rebrickable colors.csv export, plus CIELAB
nearest-colour matching so photo/mesh colours can be snapped to real brick
colours.

The Rebrickable bulk colors.csv download only has id,name,rgb,is_trans;
the LDraw/BrickLink id mapping comes from their API and gets merged in by
whoever produces the file placed at data/colors.csv, so the exact shape of
that mapping varies. load_colours() accepts either an "external_ids" column
(JSON, or a Python-repr dict if it was dumped with str() instead of
json.dumps()) or flat "ldraw_id"/"bricklink_id" columns.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "colors.csv"

PALETTE_NAMES = [
    "Black",
    "White",
    "Red",
    "Blue",
    "Yellow",
    "Green",
    "Dark Bluish Gray",
    "Light Bluish Gray",
    "Tan",
    "Dark Tan",
    "Reddish Brown",
    "Orange",
    "Lime",
    "Dark Red",
    "Dark Blue",
    "Dark Green",
    "Sand Green",
    "Medium Azure",
    "Bright Pink",
    "Dark Orange",
]

ColourSpace = Literal["lab", "rgb"]


@dataclass(frozen=True)
class Colour:
    ldraw_id: int
    bricklink_id: int
    name: str
    rgb: tuple[int, int, int]


def _parse_rgb(value: str) -> tuple[int, int, int]:
    hex_value = value.strip().lstrip("#")
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )


def _parse_external_ids(raw: str) -> dict[str, int]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = ast.literal_eval(raw)
    ids: dict[str, int] = {}
    for key in ("LDraw", "BrickLink"):
        entry = data.get(key)
        if entry and entry.get("ext_ids"):
            ids[key] = int(entry["ext_ids"][0])
    return ids


def load_colours(csv_path: Path = DATA_PATH) -> list[Colour]:
    """Parse a colors.csv file into Colours, dropping any colour with no
    BrickLink id (and logging a warning for each one dropped)."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        has_external_ids = "external_ids" in fieldnames
        ldraw_key = next((f for f in fieldnames if f.lower() in ("ldraw_id", "ldraw")), None)
        bricklink_key = next(
            (f for f in fieldnames if f.lower() in ("bricklink_id", "bricklink")), None
        )
        if not has_external_ids and not (ldraw_key and bricklink_key):
            raise ValueError(
                f"colors.csv at {csv_path} has an unrecognised schema: {fieldnames}"
            )

        colours: list[Colour] = []
        for row in reader:
            name = row["name"]
            rgb = _parse_rgb(row["rgb"])

            if has_external_ids:
                ids = _parse_external_ids(row.get("external_ids") or "{}")
                ldraw_id = ids.get("LDraw")
                bricklink_id = ids.get("BrickLink")
            else:
                assert ldraw_key is not None and bricklink_key is not None
                ldraw_id = int(row[ldraw_key]) if row.get(ldraw_key) else None
                bricklink_id = int(row[bricklink_key]) if row.get(bricklink_key) else None

            if ldraw_id is None:
                continue
            if bricklink_id is None:
                logger.warning("colour %r has no BrickLink id; dropping", name)
                continue
            colours.append(Colour(ldraw_id=ldraw_id, bricklink_id=bricklink_id, name=name, rgb=rgb))

        return colours


def build_palette(colours: list[Colour], names: list[str] = PALETTE_NAMES) -> list[Colour]:
    """Restrict and order colours to the given list of names, skipping (and
    warning about) any name not present in colours."""
    by_name = {c.name: c for c in colours}
    palette: list[Colour] = []
    for name in names:
        colour = by_name.get(name)
        if colour is None:
            logger.warning("palette colour %r not found in colour data", name)
            continue
        palette.append(colour)
    return palette


def _load_palette() -> list[Colour]:
    if not DATA_PATH.exists():
        logger.warning(
            "colour data file missing at %s; PALETTE is empty until it is placed there", DATA_PATH
        )
        return []
    return build_palette(load_colours(DATA_PATH))


PALETTE: list[Colour] = _load_palette()


# sRGB (D65) -> CIE XYZ, and the D65 reference white used to normalise it.
_XYZ_MATRIX = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE = (0.95047, 1.00000, 1.08883)
_DELTA = 6.0 / 29.0


def srgb_array_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorised sRGB (D65) -> CIELAB conversion for an (..., 3) uint8/int
    array, so we don't need a colour-science dependency for batch
    nearest-colour matching across many voxels at once."""
    v = rgb.astype(np.float64) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    matrix = np.array(_XYZ_MATRIX)
    xyz = linear @ matrix.T
    xyz = xyz / np.array(_D65_WHITE)

    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4.0 / 29.0)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Hand-rolled sRGB (D65) -> CIELAB conversion, so we don't need a
    colour-science dependency just for nearest-colour matching."""
    lab = srgb_array_to_lab(np.array(rgb))
    return float(lab[0]), float(lab[1]), float(lab[2])


def nearest_colour(
    rgb: tuple[int, int, int], palette: list[Colour], space: ColourSpace = "lab"
) -> Colour:
    """Return the palette Colour closest to rgb by Euclidean distance in the
    given colour space."""
    if space == "lab":
        target = srgb_to_lab(rgb)

        def distance(colour: Colour) -> float:
            other = srgb_to_lab(colour.rgb)
            return sum((a - b) ** 2 for a, b in zip(target, other, strict=True))

    elif space == "rgb":

        def distance(colour: Colour) -> float:
            return sum((a - b) ** 2 for a, b in zip(rgb, colour.rgb, strict=True))

    else:
        raise ValueError(f"unknown colour space: {space!r}")

    return min(palette, key=distance)


def nearest_colour_batch(
    rgb: np.ndarray, palette: list[Colour], space: ColourSpace = "lab"
) -> np.ndarray:
    """Vectorised sibling of nearest_colour: rgb is (N, 3) uint8/int: returns
    an (N,) int array of the ldraw_id of each nearest palette Colour."""
    if not palette:
        raise ValueError("palette must not be empty")

    palette_rgb = np.array([c.rgb for c in palette])
    ldraw_ids = np.array([c.ldraw_id for c in palette])

    if space == "lab":
        sample = srgb_array_to_lab(rgb)
        targets = srgb_array_to_lab(palette_rgb)
    elif space == "rgb":
        sample = rgb.astype(np.float64)
        targets = palette_rgb.astype(np.float64)
    else:
        raise ValueError(f"unknown colour space: {space!r}")

    distances = np.sum((sample[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
    nearest_index = np.argmin(distances, axis=1)
    return np.asarray(ldraw_ids[nearest_index])
