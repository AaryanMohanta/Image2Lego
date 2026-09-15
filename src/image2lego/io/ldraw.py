"""Write and read LDraw (.ldr) files for a brick layout.

Coordinate convention
----------------------
LDraw is a right-handed system with -Y pointing *up*. A Brick's (x, y, z)
is its min-corner stud position, with y counting layers *up* from 0 at
ground level (the opposite sense of LDraw's Y axis) and (x, z) the stud
grid in the horizontal plane.

Each part's origin in its .dat file is at the top-centre of its body (the
underside of its studs), which is the standard LDraw convention for bricks
and plates. So placing a brick whose min-corner is at stud layer y means
its origin sits one brick/plate height *above* the top of that layer:

    LDU_x = (x + w/2) * STUD_LDU
    LDU_y = -(y + 1) * H          # H = BRICK_H_LDU or PLATE_H_LDU
    LDU_z = (z + d/2) * STUD_LDU

Y_ORIGIN_MODE = "top" encodes that assumption. If visual inspection in a
viewer (e.g. BrickLink Studio) shows the whole model offset by one layer,
the part origin convention was actually "bottom" (origin at the underside
resting on the plate below) and Y_ORIGIN_MODE should be flipped to "bottom",
which changes the y formula to `-y * H`.

Rotation: a placed Brick's (w, d) already reflect its footprint after
rotation (w along x, d along z). A part's .dat file is authored with its
long axis along LDraw X in its identity orientation, so a brick is
"unrotated" (identity matrix) when w >= d, and rotated 90 degrees about Y
(swapping x/z) when d > w.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from image2lego.model import BRICK_H_LDU, PLATE_H_LDU, STUD_LDU, Brick, LayerType

Y_ORIGIN_MODE = "top"

_IDENTITY_MATRIX = (1, 0, 0, 0, 1, 0, 0, 0, 1)
_ROTATED_MATRIX = (0, 0, 1, 0, 1, 0, -1, 0, 0)


def _layer_height(layer_type: LayerType) -> int:
    return BRICK_H_LDU if layer_type == "brick" else PLATE_H_LDU


def _fmt(n: float) -> str:
    if n == int(n):
        return str(int(n))
    return f"{n:.3f}".rstrip("0").rstrip(".")


def _brick_line(brick: Brick, layer_type: LayerType) -> str:
    h = _layer_height(layer_type)
    ldu_x = (brick.x + brick.w / 2) * STUD_LDU
    ldu_y = -(brick.y + 1) * h
    ldu_z = (brick.z + brick.d / 2) * STUD_LDU
    matrix = _ROTATED_MATRIX if brick.d > brick.w else _IDENTITY_MATRIX

    coords = " ".join(_fmt(n) for n in (ldu_x, ldu_y, ldu_z))
    matrix_str = " ".join(str(n) for n in matrix)
    return f"1 {brick.colour} {coords} {matrix_str} {brick.part_id}.dat"


def write_ldr(
    bricks: list[Brick], path: str | Path, layer_type: LayerType, title: str
) -> None:
    path = Path(path)
    lines = [
        f"0 {title}",
        f"0 Name: {path.name}",
        "0 Author: image2lego",
        "0 !LDRAW_ORG Model",
    ]
    lines.extend(_brick_line(brick, layer_type) for brick in bricks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LDrawRecord(NamedTuple):
    colour: int
    position: tuple[float, float, float]
    matrix: tuple[float, float, float, float, float, float, float, float, float]
    part: str


def read_ldr(path: str | Path) -> list[LDrawRecord]:
    records: list[LDrawRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "1":
            continue
        colour = int(tokens[1])
        x, y, z = (float(v) for v in tokens[2:5])
        m0, m1, m2, m3, m4, m5, m6, m7, m8 = (float(v) for v in tokens[5:14])
        matrix = (m0, m1, m2, m3, m4, m5, m6, m7, m8)
        part = tokens[14]
        records.append(LDrawRecord(colour=colour, position=(x, y, z), matrix=matrix, part=part))
    return records
