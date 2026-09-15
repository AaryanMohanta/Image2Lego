from pathlib import Path

import pytest
from PIL import Image

from image2lego.frontend.base import ImageToMesh
from image2lego.frontend.file import FileBackend
from image2lego.frontend.hosted import HostedBackend
from image2lego.frontend.trellis2 import DEFAULT_TRELLIS2_URL, Trellis2Backend


class TestImageToMeshIsAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            ImageToMesh()  # type: ignore[abstract]

    def test_subclass_without_generate_cannot_be_instantiated(self) -> None:
        class Incomplete(ImageToMesh):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestFileBackend:
    def test_generate_ignores_image_and_returns_configured_mesh(
        self, tmp_path: Path
    ) -> None:
        source_mesh = tmp_path / "source.glb"
        source_mesh.write_bytes(b"fake glb bytes")

        backend = FileBackend(source_mesh)
        out_dir = tmp_path / "out"
        image = Image.new("RGB", (4, 4), color=(255, 0, 0))

        result = backend.generate(image, out_dir, seed=0)

        assert result == out_dir / "mesh.glb"
        assert result.read_bytes() == b"fake glb bytes"

    def test_generate_creates_out_dir_if_missing(self, tmp_path: Path) -> None:
        source_mesh = tmp_path / "source.glb"
        source_mesh.write_bytes(b"x")
        backend = FileBackend(source_mesh)
        out_dir = tmp_path / "nested" / "out"
        image = Image.new("RGB", (2, 2))

        backend.generate(image, out_dir, seed=0)

        assert out_dir.exists()

    def test_different_seeds_still_return_same_mesh(self, tmp_path: Path) -> None:
        source_mesh = tmp_path / "source.glb"
        source_mesh.write_bytes(b"stable")
        backend = FileBackend(source_mesh)
        image = Image.new("RGB", (2, 2))

        a = backend.generate(image, tmp_path / "a", seed=0)
        b = backend.generate(image, tmp_path / "b", seed=99)

        assert a.read_bytes() == b.read_bytes() == b"stable"


class TestHostedBackendConstruction:
    def test_raises_without_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IMAGE2LEGO_HOSTED_BASE_URL", raising=False)
        monkeypatch.delenv("IMAGE2LEGO_HOSTED_API_KEY", raising=False)
        with pytest.raises(ValueError, match="IMAGE2LEGO_HOSTED_BASE_URL"):
            HostedBackend()

    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IMAGE2LEGO_HOSTED_API_KEY", raising=False)
        with pytest.raises(ValueError, match="IMAGE2LEGO_HOSTED_API_KEY"):
            HostedBackend(base_url="https://example.com")

    def test_constructs_from_explicit_args(self) -> None:
        backend = HostedBackend(base_url="https://example.com/", api_key="secret")
        assert backend.base_url == "https://example.com"
        assert backend._headers() == {"Authorization": "Bearer secret"}

    def test_constructs_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE2LEGO_HOSTED_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("IMAGE2LEGO_HOSTED_API_KEY", "env-secret")
        backend = HostedBackend()
        assert backend.base_url == "https://api.example.com"

    def test_custom_field_map_overrides_only_given_keys(self) -> None:
        backend = HostedBackend(
            base_url="https://example.com",
            api_key="k",
            field_map={"task_id_field": "data.id"},
        )
        assert backend.field_map["task_id_field"] == "data.id"
        assert backend.field_map["status_field"] == "status"  # default preserved

    def test_never_logs_the_api_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = HostedBackend(base_url="https://example.com", api_key="super-secret-key")

        class _FakeResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, str]:
                return {"status": "success", "model_url": "https://example.com/out.glb"}

        monkeypatch.setattr(
            "image2lego.frontend.hosted.requests.get", lambda *a, **kw: _FakeResponse()
        )

        with caplog.at_level("DEBUG"):
            backend._poll_until_done("abc123")

        assert "super-secret-key" not in caplog.text


class TestTrellis2BackendConstruction:
    def test_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRELLIS2_URL", raising=False)
        backend = Trellis2Backend()
        assert backend.base_url == DEFAULT_TRELLIS2_URL

    def test_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELLIS2_URL", "http://gpu-box:9000/")
        backend = Trellis2Backend()
        assert backend.base_url == "http://gpu-box:9000"

    def test_explicit_base_url_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRELLIS2_URL", "http://from-env:8765")
        backend = Trellis2Backend(base_url="http://explicit:1234")
        assert backend.base_url == "http://explicit:1234"
