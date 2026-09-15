"""HostedBackend: a generic image-to-3D REST API client.

Works with any service following the common "submit job -> poll status ->
download result" pattern used by hosted image-to-3D APIs (e.g. Tripo AI,
Meshy). Configuration is split in two:

- Base URL, API key, and endpoint paths come from environment variables,
  so credentials never end up in code or in this repo.
- Request/response *field names* are configurable via `field_map`, since
  providers name things differently (Tripo's task-id field isn't Meshy's),
  without needing a new backend class per provider.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from image2lego.frontend.base import ImageToMesh

logger = logging.getLogger(__name__)

# Default field names, matching a generic "submit -> poll -> glb url" shape.
# Override any of these via `field_map` for a specific provider.
DEFAULT_FIELD_MAP: dict[str, str] = {
    "image_field": "image",  # multipart field name for the uploaded image
    "task_id_field": "task_id",  # response field holding the new task id
    "status_field": "status",  # response field holding job status
    "status_done_value": "success",  # status value meaning "finished"
    "status_failed_value": "failed",  # status value meaning "errored"
    "error_field": "error",  # response field with an error message
    "model_url_field": "model_url",  # response field with the GLB URL
}

ENV_BASE_URL = "IMAGE2LEGO_HOSTED_BASE_URL"
ENV_API_KEY = "IMAGE2LEGO_HOSTED_API_KEY"
ENV_CREATE_ENDPOINT = "IMAGE2LEGO_HOSTED_CREATE_ENDPOINT"
ENV_STATUS_ENDPOINT = "IMAGE2LEGO_HOSTED_STATUS_ENDPOINT"


def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    """data["a"]["b"] via dotted_key="a.b", for providers that nest their
    responses (e.g. {"data": {"task_id": ...}})."""
    value: Any = data
    for part in dotted_key.split("."):
        value = value[part]
    return value


class HostedBackend(ImageToMesh):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        create_endpoint: str | None = None,
        status_endpoint: str | None = None,
        field_map: dict[str, str] | None = None,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL, "")).rstrip("/")
        if not self.base_url:
            raise ValueError(f"HostedBackend needs {ENV_BASE_URL} (or base_url=).")

        self._api_key = api_key or os.environ.get(ENV_API_KEY)
        if not self._api_key:
            raise ValueError(f"HostedBackend needs {ENV_API_KEY} (or api_key=).")

        self.create_endpoint = create_endpoint or os.environ.get(ENV_CREATE_ENDPOINT, "/task")
        self.status_endpoint = status_endpoint or os.environ.get(
            ENV_STATUS_ENDPOINT, "/task/{task_id}"
        )
        self.field_map = {**DEFAULT_FIELD_MAP, **(field_map or {})}
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        # Never log this -- it carries the API key.
        return {"Authorization": f"Bearer {self._api_key}"}

    def generate(self, image: Image.Image, out_dir: Path, seed: int = 0) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        processed = self.preprocess(image)

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)

        fm = self.field_map
        create_url = f"{self.base_url}{self.create_endpoint}"
        logger.info("submitting image to %s", create_url)
        response = requests.post(
            create_url,
            headers=self._headers(),
            files={fm["image_field"]: ("subject.png", buf, "image/png")},
            data={"seed": str(seed)},
            timeout=60,
        )
        response.raise_for_status()
        task_id = _get_nested(response.json(), fm["task_id_field"])

        model_url = self._poll_until_done(task_id)

        glb_response = requests.get(model_url, timeout=120)
        glb_response.raise_for_status()

        dest = out_dir / "mesh.glb"
        dest.write_bytes(glb_response.content)
        return dest

    def _poll_until_done(self, task_id: Any) -> str:
        fm = self.field_map
        status_url = f"{self.base_url}{self.status_endpoint.format(task_id=task_id)}"
        deadline = time.monotonic() + self.timeout

        while True:
            response = requests.get(status_url, headers=self._headers(), timeout=30)
            response.raise_for_status()
            payload = response.json()
            status = _get_nested(payload, fm["status_field"])

            if status == fm["status_done_value"]:
                model_url: str = _get_nested(payload, fm["model_url_field"])
                return model_url
            if status == fm["status_failed_value"]:
                try:
                    error = _get_nested(payload, fm["error_field"])
                except (KeyError, TypeError):
                    error = payload
                raise RuntimeError(f"hosted image-to-3D job {task_id} failed: {error}")

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"hosted image-to-3D job {task_id} did not finish within {self.timeout}s"
                )
            time.sleep(self.poll_interval)
