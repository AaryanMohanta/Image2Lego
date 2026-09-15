"""image2lego CLI: mesh -> voxel occupancy grid, plus a quick ASCII/PNG preview."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.text import Text

from image2lego.colours import PALETTE
from image2lego.frontend.base import ImageToMesh
from image2lego.frontend.file import FileBackend
from image2lego.frontend.hosted import HostedBackend
from image2lego.frontend.trellis2 import Trellis2Backend
from image2lego.geometry.colorize import colorize
from image2lego.geometry.mesh import load_mesh, orient_mesh, scale_mesh_to_studs
from image2lego.geometry.voxelize import summarise, voxelize
from image2lego.io.bricklink import write_wanted_list
from image2lego.io.ldraw import LDrawRecord, read_ldr, write_ldr
from image2lego.legolize import legolize
from image2lego.legolize.graph import build_graph, score, weak_points
from image2lego.legolize.greedy import greedy_all
from image2lego.legolize.repair import ground_floating
from image2lego.legolize.stability import analyse_stability
from image2lego.model import (
    BRICK_H_LDU,
    PLATE_H_LDU,
    STUD_LDU,
    Brick,
    ColourGrid,
    LayerType,
    Occupancy,
    PartCatalogue,
)
from image2lego.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

console = Console()
app = typer.Typer(help="image2lego: turn a 3D model into a LEGO brick layout.")

MAX_RECOMMENDED_BRICKS = 5000
IDEAS_MIN_ELEMENTS = 200
IDEAS_MAX_ELEMENTS = 5000


class UpChoice(StrEnum):
    auto = "auto"
    x = "x"
    y = "y"
    z = "z"


class BackendChoice(StrEnum):
    file = "file"
    hosted = "hosted"
    trellis2 = "trellis2"


class SymmetriseChoice(StrEnum):
    x = "x"
    z = "z"


@contextmanager
def _timed(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - start


def _make_backend(backend: BackendChoice, mesh: Path | None) -> ImageToMesh:
    if backend == BackendChoice.file:
        if mesh is None:
            raise typer.BadParameter("--backend file needs --mesh <path to an existing .glb>.")
        return FileBackend(mesh)
    if backend == BackendChoice.hosted:
        return HostedBackend()
    return Trellis2Backend()


@app.command("build")
def build_command(
    photo: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input product photo."),
    backend: BackendChoice = typer.Option(
        BackendChoice.file, "--backend", help="Image-to-mesh backend."
    ),
    mesh: Path | None = typer.Option(
        None, "--mesh", exists=True, dir_okay=False, help="Existing .glb, for --backend file."
    ),
    width: int = typer.Option(..., "--width", help="Width in studs (longer of the x/z extents)."),
    plates: bool = typer.Option(
        False, "--plates/--bricks", help="Use plate height instead of brick height."
    ),
    hollow: int = typer.Option(
        3, "--hollow", help="Shell thickness in voxels to hollow the model by; 0 disables it."
    ),
    up: UpChoice = typer.Option(UpChoice.auto, "--up", help="Which source axis is up."),
    symmetrise_axis: SymmetriseChoice | None = typer.Option(
        None,
        "--symmetrise",
        help="Mirror-symmetrise the voxel grid across this axis (keeping whichever "
        "half has more voxels).",
    ),
    max_colours: int | None = typer.Option(
        None, "--max-colours", help="Cluster surface colours down to at most this many."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    restarts: int = typer.Option(4, "--restarts", help="Number of greedy restarts to try."),
    repair_iters: int = typer.Option(
        200, "--repair-iters", help="Max weak-point repair iterations."
    ),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory."),
) -> None:
    """Run the full photo -> LEGO model pipeline: preprocess the photo,
    generate a mesh, orient/scale it to studs, voxelize, optionally
    mirror-symmetrise, colorize, legolize, and write model.ldr, wanted.xml,
    parts.csv, preview.png and report.json into --out. Intermediate
    artefacts (subject.png, mesh.glb, occ.npz) are also saved there, so
    any stage can be re-run from them directly."""
    active_backend = _make_backend(backend, mesh)
    layer_type: LayerType = "plate" if plates else "brick"
    image = Image.open(photo).convert("RGB")

    report = run_pipeline(
        image,
        active_backend,
        out,
        width,
        layer_type=layer_type,
        hollow=hollow,
        up=up.value,
        symmetrise_axis=symmetrise_axis.value if symmetrise_axis is not None else None,
        max_colours=max_colours,
        seed=seed,
        restarts=restarts,
        repair_iters=repair_iters,
        log=console.print,
    )

    table = Table(title="build summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("total bricks", str(report["brick_count"]))
    table.add_row("articulation points", str(report["articulation_points"]))
    table.add_row(
        f"within LEGO Ideas window ({IDEAS_MIN_ELEMENTS}-{IDEAS_MAX_ELEMENTS})",
        "yes" if report["within_lego_ideas_window"] else "no",
    )
    console.print(table)
    console.print(f"\n[bold green]Done.[/bold green] Outputs in {out}")


@app.command("voxelize")
def voxelize_command(
    mesh_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a glb/gltf/obj/stl/ply mesh."
    ),
    width: int = typer.Option(..., "--width", help="Width in studs (longer of the x/z extents)."),
    plates: bool = typer.Option(
        False, "--plates/--bricks", help="Use plate height instead of brick height."
    ),
    hollow: int = typer.Option(
        3, "--hollow", help="Shell thickness in voxels to hollow the model by; 0 disables it."
    ),
    up: UpChoice = typer.Option(UpChoice.auto, "--up", help="Which source axis is up."),
    out: Path = typer.Option(..., "--out", help="Output .npz path."),
) -> None:
    """Voxelize a mesh into an occupancy grid and save it to --out."""
    layer_type: LayerType = "plate" if plates else "brick"

    console.print(f"[bold]Loading[/bold] {mesh_path}")
    mesh = load_mesh(mesh_path)
    mesh = orient_mesh(mesh, up=up.value)
    unscaled_mesh = mesh.copy()
    mesh, transform = scale_mesh_to_studs(mesh, width_studs=width, layer_type=layer_type)

    hollow_thickness = hollow if hollow > 0 else None
    occ, offset = voxelize(mesh, hollow_thickness=hollow_thickness, keep_components="largest")
    # The pre-hollow solid shape, kept separately so `legolize` can route
    # filler columns through material that voxelize()'s hollowing removed.
    solid_occ, _solid_offset = voxelize(mesh, hollow_thickness=None, keep_components="largest")

    summary = summarise(occ)
    console.print(
        f"dims={summary['dims']} voxels={summary['voxel_count']} "
        f"layers={summary['layer_count']} est_bricks~={summary['estimated_brick_count']}"
    )

    estimated_bricks = summary["estimated_brick_count"]
    assert isinstance(estimated_bricks, int)
    if estimated_bricks > MAX_RECOMMENDED_BRICKS:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] estimated {estimated_bricks} bricks "
            f"exceeds the recommended {MAX_RECOMMENDED_BRICKS}-part limit."
        )

    if PALETTE:
        console.print("[bold]Colorizing[/bold]")
        colours = colorize(occ, unscaled_mesh, transform, PALETTE)
    else:
        console.print(
            "[yellow]No colour palette available (image2lego/data/colors.csv missing) "
            "-- skipping colour sampling.[/yellow]"
        )
        colours = np.full(occ.shape, -1, dtype=np.int16)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        occupancy=occ,
        offset=offset,
        colours=colours,
        solid_occ=solid_occ,
        layer_type=np.array(layer_type),
        width_studs=np.array(width),
    )
    console.print(f"[green]Wrote[/green] {out}")


@app.command("legolize")
def legolize_command(
    occ_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a .npz saved by `voxelize`."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed."),
    restarts: int = typer.Option(4, "--restarts", help="Number of greedy restarts to try."),
    repair_iters: int = typer.Option(
        200, "--repair-iters", help="Max weak-point repair iterations."
    ),
    out: Path = typer.Option(..., "--out", help="Output .ldr path."),
    xml: Path | None = typer.Option(
        None, "--xml", help="Output BrickLink wanted-list XML path."
    ),
) -> None:
    """Turn a voxelized/colorized .npz into a brick layout, writing an LDraw
    model (and optionally a BrickLink wanted list)."""
    data = np.load(occ_path)
    colours: ColourGrid = data["colours"]
    solid_occ: Occupancy = (
        data["solid_occ"].astype(bool) if "solid_occ" in data else data["occupancy"].astype(bool)
    )
    layer_type: LayerType = str(data["layer_type"].item())  # type: ignore[assignment]

    catalogue = PartCatalogue()

    console.print("[bold]Greedy merging + grounding[/bold] (pre-repair)")
    best_bricks = None
    best_score = float("inf")
    for i in range(restarts):
        candidate = greedy_all(colours, catalogue, layer_type, seed=seed + i)
        candidate_score = score(candidate, build_graph(candidate), (1.0, 5.0, 2.0, 0.5))
        if candidate_score < best_score:
            best_score = candidate_score
            best_bricks = candidate
    assert best_bricks is not None
    pre_repair = ground_floating(best_bricks, colours, solid_occ, catalogue, layer_type)
    before_articulation = len(
        weak_points(build_graph(pre_repair), pre_repair)["articulation_points"]
    )

    console.print("[bold]Repairing weak points[/bold]")
    bricks = legolize(
        colours,
        solid_occ,
        catalogue,
        layer_type,
        seed=seed,
        restarts=restarts,
        repair_iters=repair_iters,
    )
    after_articulation = len(weak_points(build_graph(bricks), bricks)["articulation_points"])

    part_counts: dict[str, int] = {}
    colour_counts: dict[int, int] = {}
    for brick in bricks:
        part_counts[brick.part_id] = part_counts.get(brick.part_id, 0) + 1
        colour_counts[brick.colour] = colour_counts.get(brick.colour, 0) + 1

    table = Table(title="legolize summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("total bricks", str(len(bricks)))
    for part_id, count in sorted(part_counts.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  part {part_id} ({catalogue.name(part_id)})", str(count))
    for colour_id, count in sorted(colour_counts.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  colour {colour_id}", str(count))
    table.add_row("articulation points (before repair)", str(before_articulation))
    table.add_row("articulation points (after repair)", str(after_articulation))
    in_ideas_range = IDEAS_MIN_ELEMENTS <= len(bricks) <= IDEAS_MAX_ELEMENTS
    table.add_row(
        f"within LEGO Ideas window ({IDEAS_MIN_ELEMENTS}-{IDEAS_MAX_ELEMENTS})",
        "yes" if in_ideas_range else "no",
    )
    console.print(table)

    out.parent.mkdir(parents=True, exist_ok=True)
    write_ldr(bricks, out, layer_type=layer_type, title=out.stem)
    console.print(f"[green]Wrote[/green] {out}")

    if xml is not None:
        if PALETTE:
            colour_map = {c.ldraw_id: c for c in PALETTE}
            xml.parent.mkdir(parents=True, exist_ok=True)
            paths = write_wanted_list(bricks, xml, colours=colour_map)
            for path in paths:
                console.print(f"[green]Wrote[/green] {path}")
        else:
            console.print(
                "[yellow]No colour palette available (image2lego/data/colors.csv missing) "
                "-- skipping wanted-list XML.[/yellow]"
            )


def _any_axis(occ: Occupancy, axis: int) -> np.ndarray:
    return np.asarray(occ.any(axis=axis))


def _face_colours(colours: ColourGrid, occ: Occupancy, axis: int) -> ColourGrid:
    """For each cell along the other two axes, the colour id of the
    occupied voxel with the largest index along `axis` -- i.e. the face
    you'd see looking down that axis from its positive end -- or -1 where
    nothing is occupied."""
    flipped_occ = np.flip(occ, axis=axis)
    first_true_from_far_end = np.argmax(flipped_occ, axis=axis)
    last_true_index = occ.shape[axis] - 1 - first_true_from_far_end

    face = np.take_along_axis(colours, np.expand_dims(last_true_index, axis=axis), axis=axis)
    face = np.squeeze(face, axis=axis)
    return np.where(occ.any(axis=axis), face, -1).astype(np.int16)


def _grid_to_array(grid: np.ndarray) -> np.ndarray:
    """Flip a (horizontal, vertical) grid so row 0 is the highest vertical
    index, matching how up-is-up should read on screen/in an image."""
    return np.flipud(grid.T)


def _grid_to_ascii(grid: np.ndarray) -> str:
    rows = _grid_to_array(grid)
    return "\n".join("".join("#" if cell else "." for cell in row) for row in rows)


def _colour_grid_to_rich_text(grid: ColourGrid, id_to_rgb: dict[int, tuple[int, int, int]]) -> Text:
    rows = _grid_to_array(grid)
    text = Text()
    for row in rows:
        for colour_id in row:
            colour_id = int(colour_id)
            if colour_id == -1:
                text.append("  ")
            else:
                r, g, b = id_to_rgb.get(colour_id, (128, 128, 128))
                text.append("##", style=f"rgb({r},{g},{b})")
        text.append("\n")
    return text


def _render_png(
    grids: dict[str, np.ndarray], path: Path, scale: int = 8, gap: int = 10
) -> None:
    images = []
    for grid in grids.values():
        arr = (_grid_to_array(grid) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        images.append(img)
    _paste_grid(images, path, gap)


def _render_colour_png(
    grids: dict[str, ColourGrid],
    id_to_rgb: dict[int, tuple[int, int, int]],
    path: Path,
    scale: int = 8,
    gap: int = 10,
) -> None:
    images = []
    for grid in grids.values():
        ids = _grid_to_array(grid)
        rgb_array = np.full((*ids.shape, 3), 40, dtype=np.uint8)
        for colour_id, rgb in id_to_rgb.items():
            rgb_array[ids == colour_id] = rgb
        img = Image.fromarray(rgb_array, mode="RGB")
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        images.append(img)
    _paste_grid(images, path, gap)


def _paste_grid(images: list[Image.Image], path: Path, gap: int) -> None:
    total_width = sum(img.width for img in images) + gap * (len(images) - 1)
    max_height = max(img.height for img in images)
    background = 40 if images[0].mode == "L" else (40, 40, 40)
    canvas = Image.new(images[0].mode, (total_width, max_height), color=background)

    x = 0
    for img in images:
        canvas.paste(img, (x, max_height - img.height))
        x += img.width + gap
    canvas.save(path)


@app.command("preview")
def preview_command(
    occ_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a .npz saved by `voxelize`."
    ),
    out: Path | None = typer.Option(None, "--out", help="PNG output path (default: <occ>.png)."),
) -> None:
    """Render top/front/side views of an occupancy grid (coloured, if the
    .npz has a colour grid) as text in the terminal, plus a PNG."""
    data = np.load(occ_path)
    occ: Occupancy = data["occupancy"]
    colours: ColourGrid | None = data["colours"] if "colours" in data else None
    has_colours = colours is not None and bool((colours != -1).any())

    if has_colours:
        assert colours is not None
        colour_faces: dict[str, ColourGrid] = {
            "top": _face_colours(colours, occ, 1),
            "front": _face_colours(colours, occ, 2),
            "side": _face_colours(colours, occ, 0),
        }
        id_to_rgb = {c.ldraw_id: c.rgb for c in PALETTE}

        for name, grid in colour_faces.items():
            console.print(f"\n[bold]{name.upper()}[/bold]")
            console.print(_colour_grid_to_rich_text(grid, id_to_rgb))

        png_path = out if out is not None else occ_path.with_suffix(".png")
        _render_colour_png(colour_faces, id_to_rgb, png_path)
    else:
        projections: dict[str, np.ndarray] = {
            "top": _any_axis(occ, 1),
            "front": _any_axis(occ, 2),
            "side": _any_axis(occ, 0),
        }

        for name, grid in projections.items():
            console.print(f"\n[bold]{name.upper()}[/bold]")
            console.print(_grid_to_ascii(grid))

        png_path = out if out is not None else occ_path.with_suffix(".png")
        _render_png(projections, png_path)

    console.print(f"\n[green]Wrote[/green] {png_path}")


def _bricks_from_ldr_records(
    records: list[LDrawRecord], catalogue: PartCatalogue
) -> tuple[list[Brick], LayerType]:
    """Reconstruct Brick objects (and the model's layer_type) from raw
    LDraw records, by inverting write_ldr's placement formula.

    Limited to models built from our own known brick/plate part ids in a
    uniform layer_type (all-brick or all-plate) and using exactly the two
    rotation matrices write_ldr emits (identity or the single 90-degree
    rotation) -- covers models we generated ourselves and simple
    Studio-compatible exports using the same parts/orientation convention,
    but not arbitrary LDraw content.
    """
    bricks: list[Brick] = []
    layer_type: LayerType | None = None

    for record in records:
        part_id = record.part.removesuffix(".dat")
        name = catalogue.name(part_id)
        kind, _, dims = name.partition(" ")
        record_layer_type: LayerType = "brick" if kind == "Brick" else "plate"
        if layer_type is None:
            layer_type = record_layer_type
        elif layer_type != record_layer_type:
            raise ValueError(f"mixed brick/plate parts are not supported (got {name})")

        w_str, _, d_str = dims.partition(" x ")
        w_canon, d_canon = int(w_str), int(d_str)

        h = BRICK_H_LDU if layer_type == "brick" else PLATE_H_LDU
        identity = record.matrix[0] > 0.5
        w, d = (max(w_canon, d_canon), min(w_canon, d_canon)) if identity else (
            min(w_canon, d_canon),
            max(w_canon, d_canon),
        )

        px, py, pz = record.position
        x = round(px / STUD_LDU - w / 2)
        y = round(-py / h - 1)
        z = round(pz / STUD_LDU - d / 2)

        bricks.append(
            Brick(x=x, y=y, z=z, w=w, d=d, colour=record.colour, part_id=part_id)
        )

    return bricks, (layer_type or "brick")


@app.command("check")
def check_command(
    ldr_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to an .ldr file (e.g. from BrickLink Studio)."
    ),
    t_max: float = typer.Option(
        1.0, "--t-max", help="Max stud tension, in 1x1-brick-weight units."
    ),
    s_max: float = typer.Option(
        1.0, "--s-max", help="Max stud/ground shear, in 1x1-brick-weight units."
    ),
    top_k: int = typer.Option(5, "--top-k", help="How many most-stressed bricks to list."),
) -> None:
    """Read an .ldr file and print a force-based stability report."""
    records = read_ldr(ldr_path)
    catalogue = PartCatalogue()
    bricks, layer_type = _bricks_from_ldr_records(records, catalogue)

    console.print(f"[bold]Loaded[/bold] {len(bricks)} {layer_type}s from {ldr_path}")

    result = analyse_stability(bricks, layer_type, t_max=t_max, s_max=s_max, top_k=top_k)

    table = Table(title="stability report")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("feasible", "yes" if result.feasible else "no")
    table.add_row("total slack", f"{result.total_slack:.4f}")
    for i in result.top_stressed_bricks:
        brick = bricks[i]
        table.add_row(
            f"  brick {i} at (x={brick.x}, y={brick.y}, z={brick.z})",
            f"slack={result.brick_slack[i]:.4f}",
        )
    console.print(table)

    if not result.feasible:
        console.print(
            "[bold yellow]Warning:[/bold yellow] model is not in static equilibrium "
            "under this simplified model."
        )


@app.command("serve")
def serve_command(
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
) -> None:
    """Run the web UI (image2lego/web.py) -- a demo, single-process,
    in-memory-job server; don't expose it beyond localhost as-is."""
    import uvicorn

    from image2lego.web import app as web_app

    console.print(f"[bold green]Serving[/bold green] on http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
