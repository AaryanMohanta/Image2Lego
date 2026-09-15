"""Evaluation harness: run the pipeline (FileBackend, no colour palette
needed) over every mesh in examples/meshes/ at widths 24/32/48 studs for
both brick and plate layers, and write results/summary.md with per-run
metrics.

Usage:
    uv run python scripts/evaluate.py

Colour sampling needs either a real photo or a real Rebrickable
colors.csv, neither of which this harness has reason to depend on -- it's
evaluating the geometry -> bricks -> stability pipeline, not colour
fidelity -- so every solid voxel is given a single fixed placeholder
colour instead of running colorize().
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from image2lego.frontend.file import FileBackend  # noqa: E402
from image2lego.geometry.mesh import load_mesh, orient_mesh, scale_mesh_to_studs  # noqa: E402
from image2lego.geometry.voxelize import summarise, voxelize  # noqa: E402
from image2lego.legolize import legolize  # noqa: E402
from image2lego.legolize.graph import build_graph, score, weak_points  # noqa: E402
from image2lego.legolize.greedy import greedy_all  # noqa: E402
from image2lego.legolize.repair import ground_floating  # noqa: E402
from image2lego.legolize.stability import analyse_stability  # noqa: E402
from image2lego.model import Brick, LayerType, PartCatalogue  # noqa: E402

MESHES_DIR = REPO_ROOT / "examples" / "meshes"
RESULTS_DIR = REPO_ROOT / "results"
SCRATCH_DIR = RESULTS_DIR / "_scratch"

WIDTHS = (24, 32, 48)
LAYER_TYPES: tuple[LayerType, ...] = ("brick", "plate")
RESTARTS = 4
REPAIR_ITERS = 200
SEED = 0
PLACEHOLDER_COLOUR = 4  # LDraw red; arbitrary -- just needs to be one consistent id.
WEIGHTS = (1.0, 5.0, 2.0, 0.5)

_MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".stl", ".ply")


class _FloatingDropCounter(logging.Handler):
    """Counts "dropping floating component" warnings emitted by
    repair.ground_floating during one run, without changing that
    function's public return type just to report this."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "dropping floating component" in record.getMessage():
            self.count += 1


def _best_greedy_layout(
    colours: np.ndarray, catalogue: PartCatalogue, layer_type: LayerType
) -> list[Brick]:
    best_bricks = None
    best_score = float("inf")
    for i in range(RESTARTS):
        candidate = greedy_all(colours, catalogue, layer_type, seed=SEED + i)
        candidate_score = score(candidate, build_graph(candidate), WEIGHTS)
        if candidate_score < best_score:
            best_score = candidate_score
            best_bricks = candidate
    assert best_bricks is not None
    return best_bricks


def evaluate_one(mesh_path: Path, width: int, layer_type: LayerType) -> dict[str, Any]:
    """Run one (mesh, width, layer_type) combination and return its row of
    metrics. Mirrors legolize()'s own restart/ground/repair pipeline
    (rather than calling it once end-to-end) so before-repair stats and
    the floating-dropped count -- which legolize() doesn't expose -- can
    be captured too; this duplicates a little work but keeps legolize()'s
    own tested implementation untouched."""
    timings: dict[str, float] = {}
    catalogue = PartCatalogue()

    t0 = time.perf_counter()
    backend = FileBackend(mesh_path)
    mesh_glb = backend.generate(Image.new("RGB", (1, 1)), SCRATCH_DIR, seed=SEED)
    mesh = load_mesh(mesh_glb)
    mesh = orient_mesh(mesh, up="auto")
    mesh, _transform = scale_mesh_to_studs(mesh, width_studs=width, layer_type=layer_type)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    occ, _offset = voxelize(mesh, hollow_thickness=3, keep_components="largest")
    solid_occ, _ = voxelize(mesh, hollow_thickness=None, keep_components="largest")
    timings["voxelize"] = time.perf_counter() - t0

    colours = np.where(occ, PLACEHOLDER_COLOUR, -1).astype(np.int16)
    summary = summarise(occ)
    voxel_count = summary["voxel_count"]
    assert isinstance(voxel_count, int)

    t0 = time.perf_counter()
    best_bricks = _best_greedy_layout(colours, catalogue, layer_type)

    drop_counter = _FloatingDropCounter()
    repair_logger = logging.getLogger("image2lego.legolize.repair")
    repair_logger.addHandler(drop_counter)
    try:
        pre_repair_bricks = ground_floating(
            best_bricks, colours, solid_occ, catalogue, layer_type
        )
    finally:
        repair_logger.removeHandler(drop_counter)
    timings["greedy_and_ground"] = time.perf_counter() - t0

    before_articulation = len(
        weak_points(build_graph(pre_repair_bricks), pre_repair_bricks)["articulation_points"]
    )

    t0 = time.perf_counter()
    bricks = legolize(
        colours,
        solid_occ,
        catalogue,
        layer_type,
        seed=SEED,
        restarts=RESTARTS,
        repair_iters=REPAIR_ITERS,
    )
    timings["legolize_repair"] = time.perf_counter() - t0

    wp = weak_points(build_graph(bricks), bricks)
    after_articulation = len(wp["articulation_points"])
    single_edge = len(wp["single_edge_bricks"])

    t0 = time.perf_counter()
    stability = analyse_stability(bricks, layer_type)
    timings["stability"] = time.perf_counter() - t0

    return {
        "mesh": mesh_path.name,
        "width": width,
        "layer_type": layer_type,
        "voxels": voxel_count,
        "bricks": len(bricks),
        "bricks_per_voxel": (len(bricks) / voxel_count) if voxel_count else 0.0,
        "articulation_before": before_articulation,
        "articulation_after": after_articulation,
        "single_edge_bricks": single_edge,
        "floating_dropped": drop_counter.count,
        "stability_feasible": stability.feasible,
        "total_slack": stability.total_slack,
        "timings": timings,
    }


def _format_timings(timings: dict[str, float]) -> str:
    return " ".join(f"{name}={seconds:.3f}s" for name, seconds in timings.items())


def _format_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['mesh']} "
        f"| {row['width']} "
        f"| {row['layer_type']} "
        f"| {row['voxels']} "
        f"| {row['bricks']} "
        f"| {row['bricks_per_voxel']:.3f} "
        f"| {row['articulation_before']} / {row['articulation_after']} "
        f"| {row['single_edge_bricks']} "
        f"| {row['floating_dropped']} "
        f"| {'yes' if row['stability_feasible'] else 'no'} "
        f"| {row['total_slack']:.4f} "
        f"| {_format_timings(row['timings'])} |"
    )


def _format_error_row(mesh_path: Path, width: int, layer_type: LayerType, error: Exception) -> str:
    message = str(error).replace("|", "/").replace("\n", " ")
    return (
        f"| {mesh_path.name} | {width} | {layer_type} "
        f"| ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR "
        f"| {message} |"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    mesh_paths = sorted(
        p for p in MESHES_DIR.iterdir() if p.suffix.lower() in _MESH_EXTENSIONS
    )
    if not mesh_paths:
        raise SystemExit(f"no meshes found under {MESHES_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        "| mesh | width | layer type | voxels | bricks | bricks/voxel "
        "| articulation (before/after) | single-edge | floating dropped "
        "| stability feasible | total slack | seconds/stage |"
    )
    separator = "|" + "---|" * 12

    rows: list[str] = []
    run_count = 0
    error_count = 0
    for mesh_path in mesh_paths:
        for width in WIDTHS:
            for layer_type in LAYER_TYPES:
                run_count += 1
                print(f"[{run_count}] {mesh_path.name} width={width} layer={layer_type}...")
                try:
                    row = evaluate_one(mesh_path, width, layer_type)
                    rows.append(_format_row(row))
                except Exception as exc:  # noqa: BLE001 -- keep evaluating the rest on failure
                    logging.exception(
                        "evaluate_one failed for %s width=%d layer=%s",
                        mesh_path.name,
                        width,
                        layer_type,
                    )
                    rows.append(_format_error_row(mesh_path, width, layer_type, exc))
                    error_count += 1

    lines = [
        "# image2lego evaluation summary",
        "",
        f"{len(mesh_paths)} mesh(es) x widths {list(WIDTHS)} x layer types "
        f"{list(LAYER_TYPES)} = {run_count} run(s), {error_count} error(s).",
        "",
        "Every solid voxel is given a single fixed placeholder colour (no real "
        "photo or colors.csv involved) -- this evaluates the geometry -> bricks "
        "-> stability pipeline, not colour fidelity.",
        "",
        header,
        separator,
        *rows,
        "",
    ]

    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"\nWrote {summary_path} ({run_count} runs, {error_count} errors)")


if __name__ == "__main__":
    main()
