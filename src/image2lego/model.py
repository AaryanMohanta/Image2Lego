from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

STUD_LDU = 20
BRICK_H_LDU = 24
PLATE_H_LDU = 8
STUD_TOP_LDU = 4

LayerType = Literal["brick", "plate"]
Occupancy = npt.NDArray[np.bool_]
ColourGrid = npt.NDArray[np.int16]


@dataclass(frozen=True)
class Brick:
    x: int
    y: int
    z: int
    w: int
    d: int
    colour: int
    part_id: str

    @property
    def footprint(self) -> set[tuple[int, int]]:
        return {
            (x, z)
            for x in range(self.x, self.x + self.w)
            for z in range(self.z, self.z + self.d)
        }

    def overlaps(self, other: "Brick") -> bool:
        if self.y != other.y:
            return False
        return not self.footprint.isdisjoint(other.footprint)


_PART_TABLE: dict[tuple[int, int], tuple[str, str]] = {
    (1, 1): ("3005", "3024"),
    (1, 2): ("3004", "3023"),
    (1, 3): ("3622", "3623"),
    (1, 4): ("3010", "3710"),
    (1, 6): ("3009", "3666"),
    (1, 8): ("3008", "3460"),
    (2, 2): ("3003", "3022"),
    (2, 3): ("3002", "3021"),
    (2, 4): ("3001", "3020"),
    (2, 6): ("2456", "3795"),
    (2, 8): ("3007", "3034"),
}

_LAYER_INDEX: dict[LayerType, int] = {"brick": 0, "plate": 1}


class PartCatalogue:
    def sizes(self, layer_type: LayerType) -> list[tuple[int, int]]:
        return sorted(_PART_TABLE.keys(), key=lambda wd: wd[0] * wd[1], reverse=True)

    def part(self, w: int, d: int, layer_type: LayerType) -> tuple[str, bool]:
        rotated = w > d
        key = (d, w) if rotated else (w, d)
        part_id = _PART_TABLE[key][_LAYER_INDEX[layer_type]]
        return part_id, rotated

    def name(self, part_id: str) -> str:
        for (w, d), (brick_id, plate_id) in _PART_TABLE.items():
            if part_id == brick_id:
                return f"Brick {w} x {d}"
            if part_id == plate_id:
                return f"Plate {w} x {d}"
        raise KeyError(f"unknown part_id: {part_id!r}")
