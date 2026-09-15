import csv
import re
from pathlib import Path

import pytest

from image2lego.colours import Colour
from image2lego.io.bricklink import write_parts_csv, write_wanted_list
from image2lego.model import Brick

_RED = Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(201, 26, 9))
_BLACK = Colour(ldraw_id=0, bricklink_id=11, name="Black", rgb=(5, 19, 29))
_COLOURS = {4: _RED, 0: _BLACK}


def _normalise_xml(text: str) -> str:
    return re.sub(r">\s+<", "><", text.strip())


def _bricks(part_id: str, colour: int, count: int) -> list[Brick]:
    return [
        Brick(x=i, y=0, z=0, w=1, d=1, colour=colour, part_id=part_id) for i in range(count)
    ]


class TestWriteWantedList:
    def test_matches_expected_xml_exactly(self, tmp_path: Path) -> None:
        bricks = _bricks("3001", 4, 12) + _bricks("3005", 0, 3)
        path = tmp_path / "wanted.xml"

        result = write_wanted_list(bricks, path, colours=_COLOURS)

        assert result == [path]
        expected = """
        <INVENTORY>
          <ITEM>
            <ITEMTYPE>P</ITEMTYPE>
            <ITEMID>3001</ITEMID>
            <COLOR>5</COLOR>
            <MINQTY>12</MINQTY>
            <CONDITION>N</CONDITION>
          </ITEM>
          <ITEM>
            <ITEMTYPE>P</ITEMTYPE>
            <ITEMID>3005</ITEMID>
            <COLOR>11</COLOR>
            <MINQTY>3</MINQTY>
            <CONDITION>N</CONDITION>
          </ITEM>
        </INVENTORY>
        """
        assert _normalise_xml(path.read_text()) == _normalise_xml(expected)

    def test_no_xml_declaration(self, tmp_path: Path) -> None:
        bricks = _bricks("3001", 4, 1)
        path = tmp_path / "wanted.xml"
        write_wanted_list(bricks, path, colours=_COLOURS)
        assert not path.read_text().lstrip().startswith("<?xml")

    def test_splits_into_multiple_files_over_size_limit(self, tmp_path: Path) -> None:
        bricks = [
            Brick(x=0, y=0, z=0, w=1, d=1, colour=4, part_id=f"{3000 + i}")
            for i in range(15000)
        ]
        colours = {4: _RED}
        path = tmp_path / "wanted.xml"

        result = write_wanted_list(bricks, path, colours=colours)

        assert len(result) > 1
        assert result[0].name == "wanted_part1.xml"
        assert result[1].name == "wanted_part2.xml"
        for part_path in result:
            assert part_path.stat().st_size <= 200_000

        total_items = 0
        for part_path in result:
            total_items += part_path.read_text().count("<ITEM>")
        assert total_items == 15000


class TestWriteParTsCsv:
    def test_columns_and_aggregated_rows(self, tmp_path: Path) -> None:
        bricks = _bricks("3001", 4, 12) + _bricks("3005", 0, 3)
        path = tmp_path / "parts.csv"

        result_path = write_parts_csv(bricks, path, colours=_COLOURS)

        assert result_path == path
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        assert rows[0].keys() == {
            "part_id",
            "part_name",
            "ldraw_colour",
            "bricklink_colour",
            "colour_name",
            "qty",
        }
        by_part = {row["part_id"]: row for row in rows}
        assert by_part["3001"]["part_name"] == "Brick 2 x 4"
        assert by_part["3001"]["ldraw_colour"] == "4"
        assert by_part["3001"]["bricklink_colour"] == "5"
        assert by_part["3001"]["colour_name"] == "Red"
        assert by_part["3001"]["qty"] == "12"
        assert by_part["3005"]["part_name"] == "Brick 1 x 1"
        assert by_part["3005"]["qty"] == "3"

    def test_missing_colour_raises_key_error(self, tmp_path: Path) -> None:
        bricks = _bricks("3001", 99, 1)
        path = tmp_path / "parts.csv"
        with pytest.raises(KeyError):
            write_parts_csv(bricks, path, colours=_COLOURS)
