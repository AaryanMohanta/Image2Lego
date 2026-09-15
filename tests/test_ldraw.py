import random
from pathlib import Path

from image2lego.io.ldraw import read_ldr, write_ldr
from image2lego.model import Brick


class TestWriteLdrHeader:
    def test_header_lines_are_well_formed(self, tmp_path: Path) -> None:
        path = tmp_path / "model.ldr"
        write_ldr([], path, layer_type="brick", title="My Model")
        lines = path.read_text().splitlines()
        assert lines[0] == "0 My Model"
        assert lines[1] == "0 Name: model.ldr"
        assert lines[2] == "0 Author: image2lego"
        assert lines[3] == "0 !LDRAW_ORG Model"


class TestWriteLdrBrickLine:
    def test_2x4_brick_unrotated_produces_exact_line(self, tmp_path: Path) -> None:
        # Real 3001.dat (Brick 2 x 4) is authored with its long (4-stud) side
        # along LDraw X in its default/identity orientation, so the
        # "unrotated" placement has w=4 (along x), d=2 (along z).
        brick = Brick(x=0, y=0, z=0, w=4, d=2, colour=4, part_id="3001")
        path = tmp_path / "model.ldr"
        write_ldr([brick], path, layer_type="brick", title="Test")

        lines = path.read_text().splitlines()
        brick_lines = [line for line in lines if line.startswith("1 ")]
        assert brick_lines == ["1 4 40 -24 20 1 0 0 0 1 0 0 0 1 3001.dat"]

    def test_rotated_1x4_plate_matches_formula(self, tmp_path: Path) -> None:
        brick = Brick(x=3, y=2, z=5, w=1, d=4, colour=71, part_id="3710")
        path = tmp_path / "model.ldr"
        write_ldr([brick], path, layer_type="plate", title="Test")

        stud_ldu = 20
        plate_h_ldu = 8
        ldu_x = (brick.x + brick.w / 2) * stud_ldu
        ldu_y = -(brick.y + 1) * plate_h_ldu
        ldu_z = (brick.z + brick.d / 2) * stud_ldu
        # d (4) > w (1): the long side runs along z, so the part is turned
        # 90 degrees relative to its identity orientation.
        matrix = "0 0 1 0 1 0 -1 0 0"

        def fmt(n: float) -> str:
            return str(int(n)) if n == int(n) else f"{n:.3f}".rstrip("0").rstrip(".")

        expected = (
            f"1 {brick.colour} {fmt(ldu_x)} {fmt(ldu_y)} {fmt(ldu_z)} {matrix} 3710.dat"
        )

        lines = path.read_text().splitlines()
        brick_lines = [line for line in lines if line.startswith("1 ")]
        assert brick_lines == [expected]

    def test_square_footprint_uses_identity_matrix(self, tmp_path: Path) -> None:
        brick = Brick(x=0, y=0, z=0, w=2, d=2, colour=0, part_id="3003")
        path = tmp_path / "model.ldr"
        write_ldr([brick], path, layer_type="brick", title="Test")
        lines = [line for line in path.read_text().splitlines() if line.startswith("1 ")]
        assert " 1 0 0 0 1 0 0 0 1 " in f" {lines[0]} "


class TestReadLdrRoundTrip:
    def test_round_trip_preserves_count_and_colours(self, tmp_path: Path) -> None:
        rng = random.Random(42)
        bricks: list[Brick] = []
        occupied: set[tuple[int, int, int]] = set()
        colours = [0, 1, 2, 4, 14, 15, 71, 72]
        attempts = 0
        while len(bricks) < 50 and attempts < 5000:
            attempts += 1
            x, y, z = rng.randint(0, 15), rng.randint(0, 3), rng.randint(0, 15)
            w, d = rng.choice([(1, 1), (1, 2), (2, 2), (1, 4), (2, 4)])
            candidate = Brick(
                x=x, y=y, z=z, w=w, d=d, colour=rng.choice(colours), part_id="3001"
            )
            cells = {(cx, y, cz) for cx, cz in candidate.footprint}
            if cells & occupied:
                continue
            occupied |= cells
            bricks.append(candidate)

        assert len(bricks) == 50

        path = tmp_path / "roundtrip.ldr"
        write_ldr(bricks, path, layer_type="brick", title="Roundtrip")
        records = read_ldr(path)

        assert len(records) == 50
        assert sorted(r.colour for r in records) == sorted(b.colour for b in bricks)

    def test_read_ldr_parses_position_matrix_and_part(self, tmp_path: Path) -> None:
        brick = Brick(x=0, y=0, z=0, w=4, d=2, colour=4, part_id="3001")
        path = tmp_path / "model.ldr"
        write_ldr([brick], path, layer_type="brick", title="Test")

        records = read_ldr(path)
        assert len(records) == 1
        record = records[0]
        assert record.colour == 4
        assert record.position == (40.0, -24.0, 20.0)
        assert record.matrix == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert record.part == "3001.dat"
