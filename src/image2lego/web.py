"""A tiny FastAPI demo UI for the photo -> LEGO model pipeline.

One page (inline HTML + vanilla JS, no frontend framework): upload a photo
or an existing mesh, pick width/plates-or-bricks/hollow/max-colours/
symmetrise, and submit. Each submission starts a background job (a plain
in-memory dict + a daemon thread per job -- fine for a single-process demo,
not meant to survive a restart or scale past one machine) that the page
polls for progress lines and, on completion, download links.

Run with `image2lego serve --port 8000`.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from image2lego.frontend.base import ImageToMesh
from image2lego.frontend.file import FileBackend
from image2lego.frontend.hosted import HostedBackend
from image2lego.frontend.trellis2 import Trellis2Backend
from image2lego.geometry.symmetry import SymmetryAxis
from image2lego.model import LayerType
from image2lego.pipeline import run_pipeline

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "done", "failed"]

# Output files a finished job may have -- also the whitelist for
# /jobs/{id}/files/{name}, so a filename can't be used to read anything
# outside a job's own output directory.
_OUTPUT_FILES = (
    "model.ldr",
    "wanted.xml",
    "parts.csv",
    "preview.png",
    "report.json",
    "subject.png",
    "mesh.glb",
    "occ.npz",
)


@dataclass
class Job:
    id: str
    out_dir: Path
    status: JobStatus = "pending"
    lines: list[str] = field(default_factory=list)
    report: dict[str, object] | None = None
    error: str | None = None


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_JOBS_ROOT = Path(tempfile.gettempdir()) / "image2lego_jobs"

app = FastAPI(title="image2lego")


def _make_backend(name: str, upload_path: Path) -> ImageToMesh:
    if name == "file":
        return FileBackend(upload_path)
    if name == "hosted":
        return HostedBackend()
    if name == "trellis2":
        return Trellis2Backend()
    raise ValueError(f"unknown backend: {name!r}")


def _run_job(
    job: Job,
    backend_name: str,
    upload_path: Path,
    width: int,
    layer_type: LayerType,
    hollow: int,
    symmetrise_axis: SymmetryAxis | None,
    max_colours: int | None,
) -> None:
    def log(message: str) -> None:
        with _JOBS_LOCK:
            job.lines.append(message)

    with _JOBS_LOCK:
        job.status = "running"

    try:
        backend = _make_backend(backend_name, upload_path)
        # FileBackend ignores the image entirely; for the other backends
        # the upload *is* the photo.
        image = (
            Image.new("RGB", (1, 1))
            if isinstance(backend, FileBackend)
            else Image.open(upload_path).convert("RGB")
        )

        report = run_pipeline(
            image,
            backend,
            job.out_dir,
            width,
            layer_type=layer_type,
            hollow=hollow,
            symmetrise_axis=symmetrise_axis,
            max_colours=max_colours,
            log=log,
        )
        with _JOBS_LOCK:
            job.report = report
            job.status = "done"
    except Exception as exc:  # noqa: BLE001 -- report *any* failure to the job, don't crash the thread silently
        logger.exception("job %s failed", job.id)
        log(f"ERROR: {exc}")
        with _JOBS_LOCK:
            job.error = str(exc)
            job.status = "failed"


@app.post("/jobs")
async def create_job(
    backend: str = Form(...),
    file: UploadFile = File(...),
    width: int = Form(...),
    plates: str = Form("false"),
    hollow: int = Form(3),
    symmetrise: str = Form("none"),
    max_colours: str = Form(""),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    out_dir = _JOBS_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    upload_name = "upload_" + (file.filename or "upload")
    upload_path = out_dir / upload_name
    upload_path.write_bytes(await file.read())

    job = Job(id=job_id, out_dir=out_dir)
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    layer_type: LayerType = "plate" if plates.strip().lower() == "true" else "brick"
    symmetrise_axis: SymmetryAxis | None = (
        None if symmetrise not in ("x", "z") else symmetrise  # type: ignore[assignment]
    )
    max_colours_value = int(max_colours) if max_colours.strip() else None

    thread = threading.Thread(
        target=_run_job,
        args=(
            job,
            backend,
            upload_path,
            width,
            layer_type,
            hollow,
            symmetrise_axis,
            max_colours_value,
        ),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")

        payload: dict[str, object] = {
            "status": job.status,
            "lines": list(job.lines),
            "error": job.error,
        }
        if job.status == "done":
            payload["report"] = job.report
            payload["links"] = {
                name: f"/jobs/{job_id}/files/{name}"
                for name in _OUTPUT_FILES
                if (job.out_dir / name).exists()
            }

    return JSONResponse(payload)


@app.get("/jobs/{job_id}/files/{filename}")
async def get_job_file(job_id: str, filename: str) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None or filename not in _OUTPUT_FILES:
        raise HTTPException(404, "not found")

    path = job.out_dir / filename
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>image2lego</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  form { display: grid; grid-template-columns: 10rem 1fr; gap: 0.6rem 1rem; align-items: center; }
  form label { text-align: right; }
  button { grid-column: 2; justify-self: start; padding: 0.5rem 1.2rem; font-size: 1rem; cursor: pointer; }
  #log { background: #111; color: #ddd; padding: 0.75rem; border-radius: 4px; height: 12rem; overflow-y: auto;
         font-family: ui-monospace, monospace; font-size: 0.85rem; white-space: pre-wrap; margin-top: 1rem; display: none; }
  #result { margin-top: 1rem; display: none; }
  #result img { max-width: 100%; border: 1px solid #ccc; }
  #result ul { padding-left: 1.2rem; }
  #error { color: #b00020; margin-top: 1rem; display: none; }
  .hint { color: #666; font-size: 0.85rem; grid-column: 2; margin-top: -0.4rem; }
</style>
</head>
<body>
<h1>image2lego</h1>
<p>Upload a photo (for a real image-to-3D backend) or an existing mesh (backend "file"), pick your
   LEGO options, and submit. This is a demo: jobs run in-process and are lost on restart.</p>

<form id="build-form">
  <label for="backend">Backend</label>
  <select id="backend" name="backend">
    <option value="file">file (upload a .glb mesh)</option>
    <option value="hosted">hosted (upload a photo)</option>
    <option value="trellis2">trellis2 (upload a photo)</option>
  </select>

  <label for="file">File</label>
  <input id="file" name="file" type="file" required>
  <div class="hint" id="file-hint">A .glb/.gltf/.obj/.stl/.ply mesh.</div>

  <label for="width">Width (studs)</label>
  <input id="width" name="width" type="number" value="32" min="1" required>

  <label for="plates">Layer type</label>
  <select id="plates" name="plates">
    <option value="false">Bricks</option>
    <option value="true">Plates</option>
  </select>

  <label for="hollow">Hollow thickness</label>
  <input id="hollow" name="hollow" type="number" value="3" min="0">

  <label for="max_colours">Max colours</label>
  <input id="max_colours" name="max_colours" type="number" min="1" placeholder="(no limit)">

  <label for="symmetrise">Symmetrise</label>
  <select id="symmetrise" name="symmetrise">
    <option value="none">none</option>
    <option value="x">x</option>
    <option value="z">z</option>
  </select>

  <button type="submit">Build</button>
</form>

<pre id="log"></pre>
<div id="error"></div>
<div id="result"></div>

<script>
const form = document.getElementById("build-form");
const backendSelect = document.getElementById("backend");
const fileInput = document.getElementById("file");
const fileHint = document.getElementById("file-hint");
const logEl = document.getElementById("log");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");

function updateFileHint() {
  if (backendSelect.value === "file") {
    fileInput.accept = ".glb,.gltf,.obj,.stl,.ply";
    fileHint.textContent = "A .glb/.gltf/.obj/.stl/.ply mesh.";
  } else {
    fileInput.accept = "image/*";
    fileHint.textContent = "A product photo.";
  }
}
backendSelect.addEventListener("change", updateFileHint);
updateFileHint();

let pollTimer = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.style.display = "none";
  resultEl.style.display = "none";
  resultEl.innerHTML = "";
  logEl.style.display = "block";
  logEl.textContent = "Submitting...\\n";
  if (pollTimer) clearInterval(pollTimer);

  const data = new FormData(form);
  let response;
  try {
    response = await fetch("/jobs", { method: "POST", body: data });
  } catch (err) {
    showError("Request failed: " + err);
    return;
  }
  if (!response.ok) {
    showError("Server returned " + response.status);
    return;
  }
  const { job_id } = await response.json();
  pollTimer = setInterval(() => pollJob(job_id), 1000);
  pollJob(job_id);
});

function showError(message) {
  errorEl.textContent = message;
  errorEl.style.display = "block";
}

async function pollJob(jobId) {
  let response;
  try {
    response = await fetch("/jobs/" + jobId);
  } catch (err) {
    return; // transient network hiccup; next poll will retry
  }
  if (!response.ok) return;
  const job = await response.json();

  logEl.textContent = job.lines.join("\\n");
  logEl.scrollTop = logEl.scrollHeight;

  if (job.status === "failed") {
    clearInterval(pollTimer);
    showError(job.error || "Build failed.");
  } else if (job.status === "done") {
    clearInterval(pollTimer);
    renderResult(jobId, job);
  }
}

function renderResult(jobId, job) {
  resultEl.style.display = "block";
  const links = job.links || {};
  let html = "";
  if (links["preview.png"]) {
    html += '<img src="' + links["preview.png"] + '" alt="preview">';
  }
  if (job.report) {
    html += "<p><strong>" + job.report.brick_count + "</strong> bricks, "
      + job.report.articulation_points + " articulation point(s), "
      + (job.report.within_lego_ideas_window ? "within" : "outside")
      + " the LEGO Ideas 200-5000 element window.</p>";
  }
  html += "<ul>";
  for (const [name, url] of Object.entries(links)) {
    html += '<li><a href="' + url + '" download>' + name + "</a></li>";
  }
  html += "</ul>";
  resultEl.innerHTML = html;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _PAGE
