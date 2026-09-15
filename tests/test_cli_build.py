import json
from pathlib import Path

import trimesh
from PIL import Image
from typer.testing import CliRunner

from image2lego.cli import app

runner = CliRunner()


def _make_photo(path: Path) -> None:
    Image.new("RGB", (32, 32), color=(200, 40, 40)).save(path)


def _make_mesh(path: Path) -> None:
    trimesh.creation.box(extents=(200, 80, 200)).export(path)


class TestBuildCommandWithFileBackend:
    def test_full_pipeline_produces_all_expected_outputs(self, tmp_path: Path) -> None:
        photo = tmp_path / "photo.jpg"
        mesh = tmp_path / "mesh.glb"
        out_dir = tmp_path / "out"
        _make_photo(photo)
        _make_mesh(mesh)

        result = runner.invoke(
            app,
            [
                "build",
                str(photo),
                "--backend",
                "file",
                "--mesh",
                str(mesh),
                "--width",
                "8",
                "--bricks",
                "--hollow",
                "0",
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output

        # Intermediate artefacts.
        assert (out_dir / "mesh.glb").exists()
        assert (out_dir / "occ.npz").exists()
        assert not (out_dir / "subject.png").exists()  # FileBackend skips preprocessing

        # Final outputs.
        assert (out_dir / "model.ldr").exists()
        assert (out_dir / "preview.png").exists()
        report_path = out_dir / "report.json"
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        assert "brick_count" in report
        assert "timings_seconds" in report
        assert set(report["timings_seconds"]) >= {
            "generate_mesh",
            "orient_scale",
            "voxelize",
            "legolize",
            "write_outputs",
        }
        assert "dims" in report
        assert report["dims"][0] == 8 or report["dims"][2] == 8  # width studs somewhere

    def test_no_colour_palette_still_completes_without_wanted_list(
        self, tmp_path: Path
    ) -> None:
        # data/colors.csv isn't guaranteed to be present; the pipeline must
        # degrade gracefully (0 bricks legolize()'d, no wanted.xml/parts.csv)
        # rather than crash.
        photo = tmp_path / "photo.jpg"
        mesh = tmp_path / "mesh.glb"
        out_dir = tmp_path / "out"
        _make_photo(photo)
        _make_mesh(mesh)

        result = runner.invoke(
            app,
            [
                "build",
                str(photo),
                "--backend",
                "file",
                "--mesh",
                str(mesh),
                "--width",
                "6",
                "--bricks",
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "report.json").exists()

    def test_symmetrise_option_runs_without_error(self, tmp_path: Path) -> None:
        photo = tmp_path / "photo.jpg"
        mesh = tmp_path / "mesh.glb"
        out_dir = tmp_path / "out"
        _make_photo(photo)
        _make_mesh(mesh)

        result = runner.invoke(
            app,
            [
                "build",
                str(photo),
                "--backend",
                "file",
                "--mesh",
                str(mesh),
                "--width",
                "8",
                "--bricks",
                "--hollow",
                "0",
                "--symmetrise",
                "x",
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output

    def test_file_backend_without_mesh_fails_clearly(self, tmp_path: Path) -> None:
        photo = tmp_path / "photo.jpg"
        _make_photo(photo)

        result = runner.invoke(
            app,
            [
                "build",
                str(photo),
                "--backend",
                "file",
                "--width",
                "8",
                "--out",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0
