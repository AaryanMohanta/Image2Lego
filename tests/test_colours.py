import csv
import logging
from pathlib import Path

import numpy as np
import pytest

from image2lego.colours import (
    Colour,
    build_palette,
    load_colours,
    nearest_colour,
    nearest_colour_batch,
    srgb_array_to_lab,
    srgb_to_lab,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestLoadColoursExternalIdsFormat:
    def test_parses_ldraw_and_bricklink_ids_from_json_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "colors.csv"
        _write_csv(
            csv_path,
            ["id", "name", "rgb", "is_trans", "external_ids"],
            [
                {
                    "id": "0",
                    "name": "Black",
                    "rgb": "05131D",
                    "is_trans": "f",
                    "external_ids": (
                        '{"LDraw": {"ext_ids": [0], "ext_descrs": [["Black"]]}, '
                        '"BrickLink": {"ext_ids": [11], "ext_descrs": [["Black"]]}}'
                    ),
                },
            ],
        )
        colours = load_colours(csv_path)
        assert colours == [
            Colour(ldraw_id=0, bricklink_id=11, name="Black", rgb=(0x05, 0x13, 0x1D))
        ]

    def test_handles_python_repr_style_external_ids(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "colors.csv"
        _write_csv(
            csv_path,
            ["id", "name", "rgb", "is_trans", "external_ids"],
            [
                {
                    "id": "4",
                    "name": "Red",
                    "rgb": "C91A09",
                    "is_trans": "f",
                    "external_ids": (
                        "{'LDraw': {'ext_ids': [4]}, 'BrickLink': {'ext_ids': [5]}}"
                    ),
                },
            ],
        )
        colours = load_colours(csv_path)
        assert colours == [Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(0xC9, 0x1A, 0x09))]


class TestLoadColoursFlatColumnFormat:
    def test_parses_separate_ldraw_and_bricklink_columns(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "colors.csv"
        _write_csv(
            csv_path,
            ["id", "name", "rgb", "is_trans", "ldraw_id", "bricklink_id"],
            [
                {
                    "id": "1",
                    "name": "Blue",
                    "rgb": "0055BF",
                    "is_trans": "f",
                    "ldraw_id": "1",
                    "bricklink_id": "7",
                },
            ],
        )
        colours = load_colours(csv_path)
        assert colours == [Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0x00, 0x55, 0xBF))]


class TestLoadColoursDropsMissingBrickLink:
    def test_drops_colour_without_bricklink_id_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        csv_path = tmp_path / "colors.csv"
        _write_csv(
            csv_path,
            ["id", "name", "rgb", "is_trans", "external_ids"],
            [
                {
                    "id": "9999",
                    "name": "ObscureColour",
                    "rgb": "123456",
                    "is_trans": "f",
                    "external_ids": '{"LDraw": {"ext_ids": [9999]}}',
                },
            ],
        )
        with caplog.at_level(logging.WARNING):
            colours = load_colours(csv_path)
        assert colours == []
        assert "ObscureColour" in caplog.text

    def test_unrecognised_schema_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "colors.csv"
        _write_csv(csv_path, ["id", "name"], [{"id": "0", "name": "Black"}])
        with pytest.raises(ValueError):
            load_colours(csv_path)


class TestBuildPalette:
    def test_filters_and_orders_by_requested_names(self) -> None:
        colours = [
            Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0, 85, 191)),
            Colour(ldraw_id=0, bricklink_id=11, name="Black", rgb=(5, 19, 29)),
            Colour(ldraw_id=99, bricklink_id=1, name="Unrelated", rgb=(1, 2, 3)),
        ]
        palette = build_palette(colours, names=["Black", "Blue"])
        assert [c.name for c in palette] == ["Black", "Blue"]

    def test_warns_and_skips_when_name_not_found(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        colours = [Colour(ldraw_id=0, bricklink_id=11, name="Black", rgb=(5, 19, 29))]
        with caplog.at_level(logging.WARNING):
            palette = build_palette(colours, names=["Black", "Nonexistent"])
        assert [c.name for c in palette] == ["Black"]
        assert "Nonexistent" in caplog.text


class TestSrgbToLab:
    def test_white_maps_to_lab_100_0_0(self) -> None:
        L, a, b = srgb_to_lab((255, 255, 255))
        assert L == pytest.approx(100.0, abs=0.1)
        assert a == pytest.approx(0.0, abs=0.1)
        assert b == pytest.approx(0.0, abs=0.1)

    def test_black_maps_to_lab_0_0_0(self) -> None:
        L, a, b = srgb_to_lab((0, 0, 0))
        assert L == pytest.approx(0.0, abs=0.1)
        assert a == pytest.approx(0.0, abs=0.1)
        assert b == pytest.approx(0.0, abs=0.1)

    def test_mid_gray_lightness_is_about_53_6(self) -> None:
        L, _a, _b = srgb_to_lab((128, 128, 128))
        assert L == pytest.approx(53.59, abs=0.5)


class TestNearestColour:
    _RED = Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(201, 26, 9))
    _GREEN = Colour(ldraw_id=2, bricklink_id=6, name="Green", rgb=(35, 120, 65))
    _BLUE = Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0, 85, 191))
    _PALETTE = [_RED, _GREEN, _BLUE]

    def test_finds_closest_colour_in_lab_space(self) -> None:
        assert nearest_colour((210, 30, 15), self._PALETTE, space="lab") == self._RED
        assert nearest_colour((10, 90, 200), self._PALETTE, space="lab") == self._BLUE

    def test_defaults_to_lab_space(self) -> None:
        assert nearest_colour((210, 30, 15), self._PALETTE) == self._RED

    def test_unknown_space_raises(self) -> None:
        with pytest.raises(ValueError):
            nearest_colour((0, 0, 0), self._PALETTE, space="xyz-unknown")  # type: ignore[arg-type]


class TestSrgbArrayToLab:
    def test_matches_scalar_conversion_elementwise(self) -> None:
        rgb = np.array([[255, 255, 255], [0, 0, 0], [128, 128, 128], [201, 26, 9]])
        batch = srgb_array_to_lab(rgb)
        for row, (r, g, b) in zip(batch, rgb, strict=True):
            expected = srgb_to_lab((int(r), int(g), int(b)))
            assert row[0] == pytest.approx(expected[0], abs=1e-6)
            assert row[1] == pytest.approx(expected[1], abs=1e-6)
            assert row[2] == pytest.approx(expected[2], abs=1e-6)

    def test_output_shape(self) -> None:
        rgb = np.zeros((5, 3), dtype=np.uint8)
        assert srgb_array_to_lab(rgb).shape == (5, 3)


class TestNearestColourBatch:
    _RED = Colour(ldraw_id=4, bricklink_id=5, name="Red", rgb=(201, 26, 9))
    _GREEN = Colour(ldraw_id=2, bricklink_id=6, name="Green", rgb=(35, 120, 65))
    _BLUE = Colour(ldraw_id=1, bricklink_id=7, name="Blue", rgb=(0, 85, 191))
    _PALETTE = [_RED, _GREEN, _BLUE]

    def test_returns_ldraw_ids_of_nearest_colours(self) -> None:
        rgb = np.array([[210, 30, 15], [10, 90, 200], [40, 125, 70]])
        ids = nearest_colour_batch(rgb, self._PALETTE)
        assert list(ids) == [self._RED.ldraw_id, self._BLUE.ldraw_id, self._GREEN.ldraw_id]

    def test_matches_scalar_nearest_colour(self) -> None:
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(20, 3))
        ids = nearest_colour_batch(rgb, self._PALETTE)
        for row, colour_id in zip(rgb, ids, strict=True):
            expected = nearest_colour((int(row[0]), int(row[1]), int(row[2])), self._PALETTE)
            assert colour_id == expected.ldraw_id
