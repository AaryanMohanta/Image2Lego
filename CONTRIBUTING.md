# Contributing to image2lego

This document describes the pipeline's stage boundaries -- what each stage
receives, what it hands off, and the invariants that must hold at each
handoff -- so changes to one stage don't quietly break another. If you're
changing a stage's internals but keeping its inputs/outputs the same
shape, you're safe. If you're changing what crosses a boundary, read the
relevant section below first.

See the [README](README.md#pipeline) for the overall data-flow diagram.
This document is about the *contracts* between stages, not the mechanics
inside them.

## The axis convention

Every occupancy grid (`Occupancy`, `bool`), colour grid (`ColourGrid`,
`int16`, `-1` = empty), and `Brick` in this codebase uses one consistent
convention, established in `image2lego/model.py` and used everywhere from
`geometry/voxelize.py` through `legolize/` to `render.py`:

- Grids are indexed `[x, y, z]`.
- **`y` is the layer index, counting up from `0` at the ground.** `y=0` is
  the bottom layer; increasing `y` goes up.
- `x` and `z` are the horizontal stud-grid axes.
- A `Brick`'s `(x, y, z)` is its minimum corner in that same system, with
  `w`/`d` extending in `+x`/`+z`.

**LDraw's own file format convention is the opposite of this: -Y is up,**
and part positions are in LDraw units (`STUD_LDU` studs, with separate
brick/plate heights) from a part's top-centre, not a grid index. This
conversion happens in exactly one place: the `io/ldraw.py` read/write
boundary. Every other module -- geometry, legolize, render, the web UI --
works entirely in image2lego's internal `[x, y-up, z]` convention and
must never see or produce -Y-up coordinates. If you're touching
`io/ldraw.py`, get the sign/offset conversion right there and nowhere
else; if you're touching anything else, you should not need to think
about LDraw's convention at all.

## Stage boundaries and their invariants

```
mesh (trimesh.Trimesh)
  |  geometry/mesh.py: load, orient, scale to studs
  v
oriented + scaled mesh
  |  geometry/voxelize.py: voxelize()
  v
occ, solid_occ (Occupancy)  +  optional symmetry.symmetrise()
  |  geometry/colorize.py: colorize()
  v
colours (ColourGrid)
  |  legolize/: greedy_all -> ground_floating -> repair_weak_points -> (optional) repair_stability
  v
bricks (list[Brick])
  |  io/ldraw.py, io/bricklink.py, render.py
  v
model.ldr, wanted.xml, parts.csv, preview.png
```

### mesh -> occupancy (`geometry/mesh.py`, `geometry/voxelize.py`)

**In:** a `trimesh.Trimesh` in whatever up-axis and units it was loaded
with.
**Out:** `occ`/`solid_occ`, two boolean `Occupancy` arrays on the
`[x, y-up, z]` stud grid (`occ` may be hollowed; `solid_occ` never is),
sharing one grid alignment.

Invariant this stage must preserve: **the mesh is fully re-expressed in
image2lego's `+Y`-up, stud-grid-aligned coordinate system before anything
downstream sees it.** `orient_mesh` and `scale_mesh_to_studs` are the only
places that reason about the source mesh's own (unknown, possibly
arbitrary) axes and units; every stage after this one assumes `y` already
means "layer index, counting up."

### occupancy -> colour grid (`geometry/colorize.py`, `geometry/symmetry.py`)

**In:** `occ`/`solid_occ` plus the *oriented, scaled* mesh (for surface
colour sampling).
**Out:** `colours`, an `int16` `ColourGrid` the same shape as `occ`, where
`-1` means "not part of the model" and any other value is an LDraw/
BrickLink colour id from `image2lego/colours.py`'s palette.

Invariant: **`colours != -1` must be a subset of `occ`** -- colorize can
leave cells uncoloured (if there's no `colors.csv`, see README) but must
never invent solid material `voxelize()` didn't produce. If you add
`symmetrise()` calls or other grid transforms in this stage, apply them
identically to `occ`/`solid_occ` and to whatever produces `colours`, so
they stay aligned.

### colour grid -> bricks (`legolize/`)

**In:** `colours` (`ColourGrid`) and `solid_occ` (`Occupancy`, used so
filler bricks introduced during repair can route through hollow-but-solid
interior space, not just coloured surface voxels).
**Out:** `bricks: list[Brick]`.

This is the stage with the most invariants, because it's where the model
goes from "a grid of numbers" to "a physical object that has to stand
up." `legolize()` asserts all three at the end of every call
(`_assert_invariants` in `image2lego/legolize/__init__.py`) -- if you're
changing anything in `legolize/greedy.py`, `graph.py`, `repair.py`, or
`stability.py`, these are the properties your change must not break:

1. **Full coverage.** Every voxel with `colours != -1` is covered by
   exactly one brick's footprint at its layer. This check is one-way:
   filler bricks from `ground_floating`/`repair_weak_points` legitimately
   occupy `colours == -1` cells (they route through hollow interior
   space to reach the ground), so not every covered cell needs to be
   coloured -- but every coloured cell must be covered.
2. **No overlap.** No two bricks may share a footprint cell at the same
   layer. Checked everywhere, filler bricks included -- unlike coverage,
   this has no exception.
3. **Single grounded component.** `build_graph(bricks)` (nodes = brick
   indices, edges where two bricks are on adjacent layers and their
   footprints overlap) must have exactly one connected component, and
   that component must touch `y=0`. A real LEGO model can't have a piece
   floating disconnected from the rest, and can't have a piece connected
   to the rest but not ultimately resting on the ground. (The trivial
   case of zero bricks -- an entirely uncoloured grid, e.g. because
   `colors.csv` isn't present -- passes vacuously: there's nothing to
   connect.)

If a change can produce a result that violates one of these, that's a bug
in the change, not a case to special-case around the assertion --
`_assert_invariants` is meant to catch that class of bug before it
reaches `io/ldraw.py` and produces a `.ldr` file that looks fine in
Studio's viewport but is physically nonsense.

### bricks -> output files (`io/ldraw.py`, `io/bricklink.py`, `render.py`)

**In:** `bricks: list[Brick]` plus the palette (`dict[int, Colour]`) for
colour lookups.
**Out:** `model.ldr` (LDraw, -Y-up), `wanted.xml`/`parts.csv`
(BrickLink), `preview.png` (isometric render).

Invariant: **this stage is purely a serializer.** It must not add,
remove, move, or resize bricks, and must not need to re-derive or
re-check any of the three `legolize` invariants above -- by the time
bricks reach this stage they're assumed final. The one nontrivial piece
of logic here is `io/ldraw.py`'s coordinate conversion (see "The axis
convention" above); everything else is formatting.

## Front ends (`pipeline.py`, `cli.py`, `web.py`)

`pipeline.py`'s `run_pipeline()` is the single orchestration function
that calls every stage above in order; both `cli.py`'s `build` command
and `web.py` call it rather than duplicating the sequence. If you need to
change the order of stages, or what gets passed between them, change it
in `run_pipeline()` -- don't special-case one front end to call stages
directly, or the CLI and web UI will drift apart.

## Running the checks yourself

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

All three run in CI (`.github/workflows/ci.yml`) on every push, CPU-only
-- image2lego itself never needs a GPU (see the README's Install
section); the one GPU-requiring piece, `services/trellis2/`, is a
separate Docker service outside this package and isn't part of CI.

`tests/test_determinism.py` is worth knowing about specifically: it
asserts that `legolize()` with a fixed seed is fully reproducible (same
brick list, not just the same brick *count*), and that different seeds
generally produce different layouts (greedy's randomized scan
corner/orientation and repair's randomized weak-point selection are both
seeded from the same `seed` argument, not from global RNG state). If
you add new randomness anywhere in `legolize/`, seed it from the `rng`/
`seed` already threaded through the call, not from an unseeded
`np.random` call -- otherwise this test (and reproducibility in general)
breaks.

`scripts/evaluate.py` runs the full pipeline over the meshes in
`examples/meshes/` at a few widths and layer types and writes
`results/summary.md`, a quick sanity/regression signal for brick counts,
structural weak points, and stability across a spread of shapes -- worth
re-running after a change to `legolize/` to see if the numbers moved in
the direction you expected.
