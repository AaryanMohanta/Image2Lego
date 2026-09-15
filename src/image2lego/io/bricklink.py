"""Export a brick layout as a BrickLink XML wanted list and a human-readable
parts CSV."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from image2lego.colours import Colour
from image2lego.model import Brick, PartCatalogue

_SIZE_LIMIT_BYTES = 200_000

_ROOT_OPEN = "<INVENTORY>\n"
_ROOT_CLOSE = "</INVENTORY>\n"
_ROOT_OVERHEAD = len((_ROOT_OPEN + _ROOT_CLOSE).encode("utf-8"))

_catalogue = PartCatalogue()


def _aggregate(bricks: list[Brick]) -> dict[tuple[str, int], int]:
    return dict(Counter((brick.part_id, brick.colour) for brick in bricks))


def _sorted_items(
    counts: dict[tuple[str, int], int], colours: dict[int, Colour]
) -> list[tuple[str, int, int]]:
    """Return (part_id, bricklink_colour_id, qty) sorted for deterministic output."""
    items = [
        (part_id, colours[ldraw_colour].bricklink_id, qty)
        for (part_id, ldraw_colour), qty in counts.items()
    ]
    return sorted(items, key=lambda item: (item[0], item[1]))


def _item_block(part_id: str, bricklink_colour: int, qty: int) -> str:
    return (
        f"  <ITEM>\n"
        f"    <ITEMTYPE>P</ITEMTYPE>\n"
        f"    <ITEMID>{part_id}</ITEMID>\n"
        f"    <COLOR>{bricklink_colour}</COLOR>\n"
        f"    <MINQTY>{qty}</MINQTY>\n"
        f"    <CONDITION>N</CONDITION>\n"
        f"  </ITEM>\n"
    )


def _serialise(items: list[tuple[str, int, int]]) -> bytes:
    body = "".join(_item_block(*item) for item in items)
    return (_ROOT_OPEN + body + _ROOT_CLOSE).encode("utf-8")


def _chunk_items(items: list[tuple[str, int, int]]) -> list[list[tuple[str, int, int]]]:
    """Greedily bin-pack items into chunks that each serialise under the size
    limit, in a single O(n) pass (no re-serialising earlier items)."""
    chunks: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_size = _ROOT_OVERHEAD
    for item in items:
        block_size = len(_item_block(*item).encode("utf-8"))
        if current and current_size + block_size > _SIZE_LIMIT_BYTES:
            chunks.append(current)
            current = []
            current_size = _ROOT_OVERHEAD
        current.append(item)
        current_size += block_size
    if current or not chunks:
        chunks.append(current)
    return chunks


def write_wanted_list(
    bricks: list[Brick], path: str | Path, colours: dict[int, Colour]
) -> list[Path]:
    """Aggregate bricks by (part_id, BrickLink colour) and write a BrickLink
    wanted-list XML. Splits into <stem>_part1.xml, _part2.xml, ... if the
    serialised output would exceed 200,000 bytes; returns the paths written."""
    path = Path(path)
    items = _sorted_items(_aggregate(bricks), colours)
    chunks = _chunk_items(items)

    if len(chunks) == 1:
        paths = [path]
    else:
        paths = [
            path.with_name(f"{path.stem}_part{i}{path.suffix}")
            for i in range(1, len(chunks) + 1)
        ]

    for chunk_path, chunk in zip(paths, chunks, strict=True):
        chunk_path.write_bytes(_serialise(chunk))

    return paths


def write_parts_csv(
    bricks: list[Brick], path: str | Path, colours: dict[int, Colour]
) -> Path:
    """Write a human-readable CSV of aggregated part counts."""
    path = Path(path)
    counts = _aggregate(bricks)
    rows = sorted(
        (
            (part_id, ldraw_colour, colours[ldraw_colour], qty)
            for (part_id, ldraw_colour), qty in counts.items()
        ),
        key=lambda row: (row[0], row[2].bricklink_id),
    )

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["part_id", "part_name", "ldraw_colour", "bricklink_colour", "colour_name", "qty"]
        )
        for part_id, ldraw_colour, colour, qty in rows:
            writer.writerow(
                [
                    part_id,
                    _catalogue.name(part_id),
                    ldraw_colour,
                    colour.bricklink_id,
                    colour.name,
                    qty,
                ]
            )
    return path
