"""The photo -> LEGO model pipeline: preprocess, generate a mesh, orient
and scale it to studs, voxelize, optionally mirror-symmetrise, colorize,
legolize, and write model.ldr, wanted.xml, parts.csv, preview.png and
report.json. Shared by the CLI `build` command and the web UI so the two
front ends can't drift out of sync with each other.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from image2lego.colours import PALETTE, Colour
from image2lego.frontend.base import ImageToMesh
from image2lego.frontend.file import FileBackend
from image2lego.geometry.colorize import colorize
from image2lego.geometry.mesh import UpAxis, load_mesh, orient_mesh, scale_mesh_to_studs
from image2lego.geometry.symmetry import SymmetryAxis, symmetrise
from image2lego.geometry.voxelize import summarise, voxelize
from image2lego.io.bricklink import write_parts_csv, write_wanted_list
from image2lego.io.ldraw import write_ldr
from image2lego.legolize import legolize
from image2lego.legolize.graph import build_graph, weak_points
from image2lego.model import LayerType, PartCatalogue
from image2lego.render import render_preview

MAX_RECOMMENDED_BRICKS = 5000
IDEAS_MIN_ELEMENTS = 200
IDEAS_MAX_ELEMENTS = 5000

LogFn = Callable[[str], None]


def _default_log(_message: str) -> None:
    pass


def run_pipeline(
    image: Image.Image,
    backend: ImageToMesh,
    out: Path,
    width: int,
    layer_type: LayerType = "brick",
    hollow: int = 3,
    up: UpAxis = "auto",
    symmetrise_axis: SymmetryAxis | None = None,
    max_colours: int | None = None,
    seed: int = 0,
    restarts: int = 4,
    repair_iters: int = 200,
    log: LogFn = _default_log,
) -> dict[str, Any]:
    """Run the full pipeline, writing every artefact under `out`, and
    return the same summary dict written to out/report.json."""
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    catalogue = PartCatalogue()

    if not isinstance(backend, FileBackend):
        log("Preprocessing photo")
        subject = timed("preprocess", lambda: backend.preprocess(image))
        subject.save(out / "subject.png")
        log(f"Wrote {out / 'subject.png'}")

    log(f"Generating mesh (backend={type(backend).__name__})")
    mesh_path = timed("generate_mesh", lambda: backend.generate(image, out, seed=seed))
    log(f"Wrote {mesh_path}")

    log("Orienting and scaling")

    def _orient_and_scale() -> Any:
        loaded = load_mesh(mesh_path)
        loaded = orient_mesh(loaded, up=up)
        unscaled = loaded.copy()
        loaded, transform = scale_mesh_to_studs(loaded, width_studs=width, layer_type=layer_type)
        return loaded, unscaled, transform

    loaded_mesh, unscaled_mesh, transform = timed("orient_scale", _orient_and_scale)

    log("Voxelizing")

    def _voxelize() -> Any:
        hollow_thickness = hollow if hollow > 0 else None
        occ, offset = voxelize(
            loaded_mesh, hollow_thickness=hollow_thickness, keep_components="largest"
        )
        solid_occ, _solid_offset = voxelize(
            loaded_mesh, hollow_thickness=None, keep_components="largest"
        )
        return occ, offset, solid_occ

    occ, offset, solid_occ = timed("voxelize", _voxelize)

    if symmetrise_axis is not None:
        log(f"Symmetrising across {symmetrise_axis}")

        def _symmetrise() -> Any:
            return symmetrise(occ, axis=symmetrise_axis), symmetrise(
                solid_occ, axis=symmetrise_axis
            )

        occ, solid_occ = timed("symmetrise", _symmetrise)

    summary = summarise(occ)
    estimated_bricks = summary["estimated_brick_count"]
    assert isinstance(estimated_bricks, int)
    if estimated_bricks > MAX_RECOMMENDED_BRICKS:
        log(
            f"Warning: estimated {estimated_bricks} bricks exceeds the recommended "
            f"{MAX_RECOMMENDED_BRICKS}-part limit."
        )

    if PALETTE:
        log("Colorizing")
        colours = timed(
            "colorize",
            lambda: colorize(occ, unscaled_mesh, transform, PALETTE, max_colours=max_colours),
        )
    else:
        log(
            "No colour palette available (image2lego/data/colors.csv missing) "
            "-- skipping colour sampling."
        )
        colours = np.full(occ.shape, -1, dtype=np.int16)

    occ_npz_path = out / "occ.npz"
    np.savez(
        occ_npz_path,
        occupancy=occ,
        offset=offset,
        colours=colours,
        solid_occ=solid_occ,
        layer_type=np.array(layer_type),
        width_studs=np.array(width),
    )
    log(f"Wrote {occ_npz_path}")

    log("Legolizing")
    bricks = timed(
        "legolize",
        lambda: legolize(
            colours,
            solid_occ,
            catalogue,
            layer_type,
            seed=seed,
            restarts=restarts,
            repair_iters=repair_iters,
        ),
    )

    part_counts: dict[str, int] = {}
    colour_counts: dict[int, int] = {}
    for brick in bricks:
        part_counts[brick.part_id] = part_counts.get(brick.part_id, 0) + 1
        colour_counts[brick.colour] = colour_counts.get(brick.colour, 0) + 1

    colour_map: dict[int, Colour] = {c.ldraw_id: c for c in PALETTE} if PALETTE else {}

    def _write_outputs() -> None:
        write_ldr(bricks, out / "model.ldr", layer_type=layer_type, title=out.name or "model")
        log(f"Wrote {out / 'model.ldr'}")

        if PALETTE:
            for path in write_wanted_list(bricks, out / "wanted.xml", colours=colour_map):
                log(f"Wrote {path}")
            write_parts_csv(bricks, out / "parts.csv", colours=colour_map)
            log(f"Wrote {out / 'parts.csv'}")
        else:
            log("No colour palette available -- skipping wanted.xml/parts.csv.")

        render_preview(bricks, colour_map, out / "preview.png", layer_type=layer_type)
        log(f"Wrote {out / 'preview.png'}")

    timed("write_outputs", _write_outputs)

    articulation = len(weak_points(build_graph(bricks), bricks)["articulation_points"])
    in_ideas_range = IDEAS_MIN_ELEMENTS <= len(bricks) <= IDEAS_MAX_ELEMENTS
    dims = summary["dims"]
    assert isinstance(dims, tuple)

    report: dict[str, Any] = {
        "brick_count": len(bricks),
        "part_counts": part_counts,
        "colour_counts": {str(colour_id): count for colour_id, count in colour_counts.items()},
        "articulation_points": articulation,
        "within_lego_ideas_window": in_ideas_range,
        "dims": list(dims),
        "voxel_count": summary["voxel_count"],
        "layer_count": summary["layer_count"],
        "estimated_brick_count": summary["estimated_brick_count"],
        "timings_seconds": {name: round(seconds, 3) for name, seconds in timings.items()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log(f"Wrote {out / 'report.json'}")
    log("Done.")

    return report
