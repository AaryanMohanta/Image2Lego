import time
from pathlib import Path
from typing import Any

import trimesh
from fastapi.testclient import TestClient

from image2lego.web import app

client = TestClient(app)


def _make_mesh(path: Path) -> None:
    trimesh.creation.box(extents=(200, 80, 200)).export(path)


def _wait_for_job(job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        job: dict[str, Any] = response.json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


class TestIndexPage:
    def test_serves_html_with_upload_form(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<form" in response.text
        assert 'name="width"' in response.text


class TestCreateAndPollJob:
    def test_file_backend_job_completes_successfully(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.glb"
        _make_mesh(mesh_path)

        with open(mesh_path, "rb") as f:
            response = client.post(
                "/jobs",
                data={
                    "backend": "file",
                    "width": "8",
                    "plates": "false",
                    "hollow": "0",
                    "symmetrise": "none",
                    "max_colours": "",
                },
                files={"file": ("mesh.glb", f, "model/gltf-binary")},
            )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        job = _wait_for_job(job_id)
        assert job["status"] == "done", job
        assert "report" in job
        assert "brick_count" in job["report"]
        assert len(job["lines"]) > 0

        assert "model.ldr" in job["links"]
        assert "preview.png" in job["links"]

    def test_downloaded_files_are_reachable(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.glb"
        _make_mesh(mesh_path)

        with open(mesh_path, "rb") as f:
            response = client.post(
                "/jobs",
                data={
                    "backend": "file",
                    "width": "8",
                    "plates": "false",
                    "hollow": "0",
                    "symmetrise": "none",
                    "max_colours": "",
                },
                files={"file": ("mesh.glb", f, "model/gltf-binary")},
            )
        job_id = response.json()["job_id"]
        job = _wait_for_job(job_id)

        ldr_url = job["links"]["model.ldr"]
        ldr_response = client.get(ldr_url)
        assert ldr_response.status_code == 200
        assert b"0 !LDRAW_ORG Model" in ldr_response.content

    def test_unknown_job_id_returns_404(self) -> None:
        response = client.get("/jobs/does-not-exist")
        assert response.status_code == 404

    def test_disallowed_filename_returns_404(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.glb"
        _make_mesh(mesh_path)
        with open(mesh_path, "rb") as f:
            response = client.post(
                "/jobs",
                data={
                    "backend": "file",
                    "width": "8",
                    "plates": "false",
                    "hollow": "0",
                    "symmetrise": "none",
                    "max_colours": "",
                },
                files={"file": ("mesh.glb", f, "model/gltf-binary")},
            )
        job_id = response.json()["job_id"]
        _wait_for_job(job_id)

        response = client.get(f"/jobs/{job_id}/files/../../etc/passwd")
        assert response.status_code == 404

    def test_symmetrise_and_max_colours_options_are_accepted(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.glb"
        _make_mesh(mesh_path)
        with open(mesh_path, "rb") as f:
            response = client.post(
                "/jobs",
                data={
                    "backend": "file",
                    "width": "8",
                    "plates": "true",
                    "hollow": "0",
                    "symmetrise": "x",
                    "max_colours": "3",
                },
                files={"file": ("mesh.glb", f, "model/gltf-binary")},
            )
        job_id = response.json()["job_id"]
        job = _wait_for_job(job_id)
        assert job["status"] == "done", job
