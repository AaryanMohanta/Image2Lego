"""Local re-optimisation: dissolve a region and re-merge it, ground floating
debris with a filler column, and hill-climb weak points (articulation
points, degree-1 bricks, repeated seams) via local dissolve/re-greedy/accept
passes.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np

from image2lego.legolize.graph import (
    WeakPoints,
    build_graph,
    grounded_components,
    score,
    weak_points,
)
from image2lego.legolize.greedy import greedy_layer
from image2lego.model import Brick, ColourGrid, LayerType, Occupancy, PartCatalogue

logger = logging.getLogger(__name__)

FILLER_COLOUR = 71  # LDraw Light_Bluish_Gray, used for purely structural fill.

_DEFAULT_WEIGHTS = (1.0, 5.0, 2.0, 0.5)

Region = tuple[int, int, int, int, int, int]  # (y0, y1, x0, x1, z0, z1)


def dissolve(bricks: list[Brick], region: Region) -> tuple[list[Brick], ColourGrid]:
    """Remove every brick whose footprint/layer intersects `region` at all
    (the whole brick is dropped, never clipped), and return the remaining
    bricks plus a colour grid -- sized exactly to `region` -- of the
    removed bricks' cells (cells of a removed brick that fall outside
    `region` are not represented; callers that need to avoid losing that
    data should widen `region` first so no removed brick's footprint
    crosses its boundary)."""
    y0, y1, x0, x1, z0, z1 = region
    remaining: list[Brick] = []
    removed: list[Brick] = []

    for brick in bricks:
        intersects = (
            y0 <= brick.y < y1
            and brick.x < x1
            and brick.x + brick.w > x0
            and brick.z < z1
            and brick.z + brick.d > z0
        )
        (removed if intersects else remaining).append(brick)

    shape = (max(x1 - x0, 0), max(y1 - y0, 0), max(z1 - z0, 0))
    grid: ColourGrid = np.full(shape, -1, dtype=np.int16)
    for brick in removed:
        ly = brick.y - y0
        if not (0 <= ly < shape[1]):
            continue
        for x, z in brick.footprint:
            lx, lz = x - x0, z - z0
            if 0 <= lx < shape[0] and 0 <= lz < shape[2]:
                grid[lx, ly, lz] = brick.colour

    return remaining, grid


def expand_region_to_cover_intersecting_bricks(bricks: list[Brick], region: Region) -> Region:
    """Grow `region` (only in x/z; bricks are always exactly one layer tall
    so y never needs to grow) until every brick that intersects it is fully
    contained, so dissolve() never has to drop data for a brick that pokes
    outside the window.

    Each round only discovers bricks touching *that round's* starting
    bounds, so a long chain of bricks overlapping one another one at a
    time (common in a densely-tiled row of a large model) can need one
    round per link in the chain to fully absorb -- the iteration cap must
    therefore scale with the input, not be a small fixed constant (an
    earlier fixed cap of 8 was observed, on a real 48-stud-wide model, to
    stop short and return a region that still had bricks poking outside
    it, silently losing their colour data in dissolve()). `len(bricks) +
    1` rounds is always enough: each round that doesn't return either
    absorbs at least one previously-untouched brick or reaches a fixed
    point, and there are only `len(bricks)` bricks to absorb."""
    y0, y1, x0, x1, z0, z1 = region
    for _ in range(len(bricks) + 1):
        touched = [
            b
            for b in bricks
            if y0 <= b.y < y1 and b.x < x1 and b.x + b.w > x0 and b.z < z1 and b.z + b.d > z0
        ]
        if not touched:
            return (y0, y1, x0, x1, z0, z1)
        new_x0 = min([x0] + [b.x for b in touched])
        new_x1 = max([x1] + [b.x + b.w for b in touched])
        new_z0 = min([z0] + [b.z for b in touched])
        new_z1 = max([z1] + [b.z + b.d for b in touched])
        if (new_x0, new_x1, new_z0, new_z1) == (x0, x1, z0, z1):
            return (y0, y1, x0, x1, z0, z1)
        x0, x1, z0, z1 = new_x0, new_x1, new_z0, new_z1
    return (y0, y1, x0, x1, z0, z1)


def _weak_point_centre(
    wp: WeakPoints, bricks: list[Brick], rng: np.random.Generator
) -> tuple[int, int, int] | None:
    articulation_points = wp["articulation_points"]
    single_edge_bricks = wp["single_edge_bricks"]
    seams = wp["seams"]

    if articulation_points:
        brick = bricks[articulation_points[int(rng.integers(0, len(articulation_points)))]]
        return (brick.x + brick.w // 2, brick.y, brick.z + brick.d // 2)
    if single_edge_bricks:
        brick = bricks[single_edge_bricks[int(rng.integers(0, len(single_edge_bricks)))]]
        return (brick.x + brick.w // 2, brick.y, brick.z + brick.d // 2)
    if seams:
        seam = seams[int(rng.integers(0, len(seams)))]
        x, y, z = seam[len(seam) // 2]
        return (x, y, z)
    return None


def _disconnection_points(
    G: nx.Graph[int], bricks: list[Brick]
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """If the model is more than one connected component (e.g. two
    separately-grounded towers that never happen to overlap on any layer --
    not necessarily a straight repeated seam), return the two closest
    brick-centre points, one from the largest component and one from the
    next largest, so a dissolve/re-greedy pass between them has a chance
    to bridge the two. Returned as (main_point, other_point) -- callers
    that need a single centre for windowing can average them; the
    `other_point` on its own is also a natural seed cell for a bridging
    placement, since it belongs to the disconnected side rather than to
    an arbitrary midpoint that may not fall on either component."""
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return None

    main, other = components[0], components[1]
    main_centres = [
        (bricks[i].x + bricks[i].w // 2, bricks[i].y, bricks[i].z + bricks[i].d // 2)
        for i in main
    ]
    other_centres = [
        (bricks[i].x + bricks[i].w // 2, bricks[i].y, bricks[i].z + bricks[i].d // 2)
        for i in other
    ]

    best: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    best_distance = math.inf
    for c1 in main_centres:
        for c2 in other_centres:
            distance = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])
            if distance < best_distance:
                best_distance = distance
                best = (c1, c2)
    assert best is not None
    return best


def repair_weak_points(
    bricks: list[Brick],
    colours: ColourGrid,
    solid_occ: Occupancy,
    catalogue: PartCatalogue,
    layer_type: LayerType,
    rng: np.random.Generator,
    max_iters: int = 200,
    window: tuple[int, int, int] = (2, 6, 6),
) -> list[Brick]:
    """Repeatedly dissolve a small window around the worst weak point
    (articulation points, then degree-1 bricks, then seams, then --
    lowest priority, since the other three usually also fire alongside it
    -- outright disconnection between separately-grounded components) and
    re-run greedy_layer over it, keeping the change only if the overall
    score improves."""
    dy, dx, dz = window
    width, height, depth = solid_occ.shape

    accepted = 0
    rejected = 0

    for _ in range(max_iters):
        graph = build_graph(bricks)

        # Disconnection is checked ahead of the three local weak-point
        # categories: a plain single_edge_brick or articulation point is
        # common (and often unfixable in isolation) in a sparse structure,
        # so if it were checked first it would starve disconnection-repair
        # every iteration even though restoring one grounded component is
        # the hard invariant repair actually has to satisfy.
        fixing_disconnection = False
        seed_point: tuple[int, int, int] | None = None
        seed_bias_point: tuple[int, int, int] | None = None
        centre: tuple[int, int, int] | None
        disconnection = _disconnection_points(graph, bricks)
        if disconnection is not None:
            fixing_disconnection = True
            main_point, other_point = disconnection
            seed_bias_point = main_point
            centre = (
                (main_point[0] + other_point[0]) // 2,
                (main_point[1] + other_point[1]) // 2,
                (main_point[2] + other_point[2]) // 2,
            )
            # Seed the bridging placement from the *disconnected* side's
            # own point, not the midpoint: the midpoint can land on a
            # cell that's already part of the main component (e.g. for
            # two touching-but-non-overlapping bricks, integer-averaging
            # their centres rounds toward whichever is already
            # connected), in which case most of the "largest brick
            # containing it" candidate placements never happen to reach
            # back out to the disconnected side at all.
            seed_point = other_point
        else:
            wp = weak_points(graph, bricks)
            centre = _weak_point_centre(wp, bricks, rng)
        if centre is None:
            break

        cx, cy, cz = centre
        nominal_y0 = max(cy - dy // 2, 0)
        nominal_y1 = min(cy - dy // 2 + dy, height)
        if fixing_disconnection and seed_point is not None and seed_bias_point is not None:
            # A plain dy-tall window centred on the midpoint of
            # main_point.y and other_point.y can round toward main_point
            # and end up excluding other_point.y entirely when the two
            # are only 1 layer apart (integer-averaging two adjacent
            # layers biases toward the lower one) -- and seed_point *is*
            # other_point, so losing it here silently disables seeding
            # for the rest of this attempt, falling back to the
            # near-zero-success unseeded scan. Widen the window to
            # guarantee both endpoints' layers are included.
            nominal_y0 = min(nominal_y0, seed_point[1], seed_bias_point[1])
            nominal_y1 = max(nominal_y1, seed_point[1] + 1, seed_bias_point[1] + 1)
        nominal = (
            nominal_y0,
            nominal_y1,
            max(cx - dx // 2, 0),
            min(cx - dx // 2 + dx, width),
            max(cz - dz // 2, 0),
            min(cz - dz // 2 + dz, depth),
        )
        region = expand_region_to_cover_intersecting_bricks(bricks, nominal)
        y0, y1, x0, x1, z0, z1 = region

        old_score = score(bricks, graph, _DEFAULT_WEIGHTS)
        old_components = nx.number_connected_components(graph)

        remaining, subgrid = dissolve(bricks, region)

        # When repairing a disconnection, bias the re-merge at the seed
        # point's own layer: greedy_layer's largest-rectangle-first scan
        # has no notion of cross-layer connectivity, so for a shape where
        # one big rectangle can be placed several ways, it will
        # consistently strand a local protrusion in its own sliver --
        # disconnected from the layer below -- regardless of scan corner,
        # since no candidate rectangle it considers is required to
        # contain the seed point specifically. A seeded placement there
        # (the largest brick that *does* contain it) is what actually has
        # a chance to bridge the two components.
        seed_local_y: int | None = None
        seed_cell_global: tuple[int, int] | None = None
        seed_bias_cell_global: tuple[int, int] | None = None
        if seed_point is not None and (
            y0 <= seed_point[1] < y1 and x0 <= seed_point[0] < x1 and z0 <= seed_point[2] < z1
        ):
            seed_local_y = seed_point[1] - y0
            seed_cell_global = (seed_point[0] - x0, seed_point[2] - z0)
            if seed_bias_point is not None:
                seed_bias_cell_global = (seed_bias_point[0] - x0, seed_bias_point[2] - z0)

        new_bricks: list[Brick] = []
        for local_y in range(subgrid.shape[1]):
            use_seed = local_y == seed_local_y
            seed_cell = seed_cell_global if use_seed else None
            seed_bias_cell = seed_bias_cell_global if use_seed else None
            layer_bricks = greedy_layer(
                subgrid[:, local_y, :], y0 + local_y, catalogue, layer_type, rng,
                seed_cell=seed_cell, seed_bias_cell=seed_bias_cell,
            )
            for b in layer_bricks:
                new_bricks.append(
                    Brick(
                        x=b.x + x0, y=b.y, z=b.z + z0, w=b.w, d=b.d, colour=b.colour,
                        part_id=b.part_id,
                    )
                )

        candidate = remaining + new_bricks
        candidate_graph = build_graph(candidate)
        new_components = nx.number_connected_components(candidate_graph)

        if fixing_disconnection:
            improved = new_components < old_components
        else:
            # A regular (non-disconnection) weak-point fix is scored on
            # score() alone, which has no connectivity term -- so without
            # this guard, a candidate that locally improves articulation/
            # seam score but happens to also sever a component (e.g. by
            # replacing a bridging brick a *previous* disconnection-repair
            # iteration created) would be accepted, silently re-breaking
            # a disconnection this same loop already fixed.
            new_score = score(candidate, candidate_graph, _DEFAULT_WEIGHTS)
            improved = new_score < old_score and new_components <= old_components

        if improved:
            bricks = candidate
            accepted += 1
        else:
            rejected += 1

    logger.info("repair_weak_points: %d accepted, %d rejected", accepted, rejected)
    return bricks


def _build_column(
    x: int,
    z: int,
    top_y: int,
    solid_occ: Occupancy,
    all_occupied: set[tuple[int, int, int]],
    grounded_occupied: set[tuple[int, int, int]],
    catalogue: PartCatalogue,
    layer_type: LayerType,
) -> list[Brick] | None:
    part_1x1, _ = catalogue.part(1, 1, layer_type)
    part_1x2, _ = catalogue.part(1, 2, layer_type)
    width, _height, depth = solid_occ.shape

    column: list[Brick] = []
    local_occupied: set[tuple[int, int, int]] = set()
    y = top_y - 1
    while y >= 0:
        if not solid_occ[x, y, z]:
            return None
        if (y, x, z) in all_occupied or (y, x, z) in local_occupied:
            return None

        part_id, w, d, bx, bz = part_1x1, 1, 1, x, z
        placed_cells = {(y, x, z)}
        for ddx, ddz, ww, dd in ((1, 0, 2, 1), (0, 1, 1, 2)):
            nx_, nz_ = x + ddx, z + ddz
            if (
                0 <= nx_ < width
                and 0 <= nz_ < depth
                and solid_occ[nx_, y, nz_]
                and (y, nx_, nz_) not in all_occupied
                and (y, nx_, nz_) not in local_occupied
            ):
                part_id, w, d, bx, bz = part_1x2, ww, dd, min(x, nx_), min(z, nz_)
                placed_cells.add((y, nx_, nz_))
                break

        column.append(Brick(x=bx, y=y, z=bz, w=w, d=d, colour=FILLER_COLOUR, part_id=part_id))
        local_occupied |= placed_cells

        if y == 0 or (y - 1, x, z) in grounded_occupied:
            return column
        y -= 1

    return None


def ground_floating(
    bricks: list[Brick],
    colours: ColourGrid,
    solid_occ: Occupancy,
    catalogue: PartCatalogue,
    layer_type: LayerType,
) -> list[Brick]:
    """For each floating connected component, try a vertical filler column
    (1x1, widened to 1x2 where a second free solid cell is available)
    straight down from a cell of its lowest brick to the ground or an
    already-grounded brick, routing only through solid_occ. Components with
    no such path are dropped (and logged) rather than left floating.

    Grounding is attempted in repeated passes rather than one linear sweep:
    a component's straight-down column can be blocked by another
    component that is itself still floating (e.g. two components stacked
    in the same column, the upper one's path passing through the lower
    one's own footprint) -- grounding that blocker in an earlier pass
    clears the path. Only components still floating after a pass makes no
    further progress are actually dropped."""
    graph = build_graph(bricks)
    grounded, floating = grounded_components(graph, bricks)

    all_occupied = {(b.y, x, z) for b in bricks for x, z in b.footprint}
    kept = [b for i, b in enumerate(bricks) if i in grounded]
    grounded_occupied = {(b.y, x, z) for b in kept for x, z in b.footprint}

    remaining = list(floating)
    while remaining:
        still_floating: list[set[int]] = []
        made_progress = False

        for component in remaining:
            comp_bricks = [bricks[i] for i in component]
            lowest_y = min(b.y for b in comp_bricks)
            candidates = [
                (x, z) for b in comp_bricks if b.y == lowest_y for x, z in b.footprint
            ]

            column: list[Brick] | None = None
            for x, z in candidates:
                column = _build_column(
                    x,
                    z,
                    lowest_y,
                    solid_occ,
                    all_occupied,
                    grounded_occupied,
                    catalogue,
                    layer_type,
                )
                if column is not None:
                    break

            if column is None:
                still_floating.append(component)
                continue

            made_progress = True
            kept.extend(comp_bricks)
            kept.extend(column)
            for b in comp_bricks + column:
                for x, z in b.footprint:
                    all_occupied.add((b.y, x, z))
                    grounded_occupied.add((b.y, x, z))

        if not made_progress:
            for component in still_floating:
                logger.warning(
                    "dropping floating component of %d brick(s): no path to ground found",
                    len(component),
                )
            break

        remaining = still_floating

    return kept
