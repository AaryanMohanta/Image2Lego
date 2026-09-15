"""Turn a coloured voxel grid into a physically sound LEGO brick layout:
greedy layer merging (with restarts), grounding floating debris, and
hill-climbing structural weak points.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from image2lego.legolize.graph import build_graph, score
from image2lego.legolize.greedy import greedy_all
from image2lego.legolize.repair import ground_floating, repair_weak_points
from image2lego.legolize.stability import repair_stability
from image2lego.model import Brick, ColourGrid, LayerType, Occupancy, PartCatalogue

Weights = tuple[float, float, float, float]

_DEFAULT_WEIGHTS: Weights = (1.0, 5.0, 2.0, 0.5)


def _assert_invariants(bricks: list[Brick], colours: ColourGrid) -> None:
    # Filler bricks from ground_floating legitimately occupy cells where
    # colours == -1 (they route through hollow-but-solid interior space),
    # so coverage is checked one-way: every *coloured* voxel must be
    # covered, not that every covered voxel must be coloured. Overlap is
    # checked everywhere, filler included.
    covered = np.zeros(colours.shape, dtype=np.int32)
    for brick in bricks:
        for x, z in brick.footprint:
            covered[x, brick.y, z] += 1

    if (covered > 1).any():
        raise AssertionError("legolize: overlapping bricks detected")
    coloured = colours != -1
    if not np.all(covered[coloured] == 1):
        raise AssertionError("legolize: not every coloured voxel is covered exactly once")

    if not bricks:
        # Coverage passing above with zero bricks means colours had no
        # coloured voxels at all -- nothing to build, which trivially
        # satisfies "one grounded component" (there's nothing to connect).
        return

    graph = build_graph(bricks)
    components = list(nx.connected_components(graph))
    grounded_components_list = [c for c in components if any(bricks[i].y == 0 for i in c)]
    if len(components) != 1 or len(grounded_components_list) != 1:
        raise AssertionError(
            f"legolize: expected exactly one grounded connected component; found "
            f"{len(components)} component(s), {len(grounded_components_list)} grounded"
        )


def legolize(
    colours: ColourGrid,
    solid_occ: Occupancy,
    catalogue: PartCatalogue,
    layer_type: LayerType,
    seed: int = 0,
    restarts: int = 4,
    repair_iters: int = 200,
    weights: Weights = _DEFAULT_WEIGHTS,
    stability: bool = False,
    stability_iters: int = 50,
    t_max: float = 1.0,
    s_max: float = 1.0,
) -> list[Brick]:
    """Run greedy_all with `restarts` different seeds and keep the
    best-scoring layout, hill-climb structural weak points (including
    reconnecting any disconnected fragments in place), then ground
    whatever debris is still floating after that. If stability, also run
    a force-based static equilibrium analysis (stability.analyse_stability)
    after that and hill-climb its most-stressed brick the same way. Raises
    AssertionError if the result fails the coverage/overlap/single-
    grounded-component invariants (this is checked regardless of
    `stability`; the stability analysis is a physical-plausibility check
    on top, not a substitute for it).

    repair_weak_points runs before ground_floating, not after: its
    disconnection-repair can reconnect a fragment in place by re-shaping
    the local brick partition (e.g. a shell bulge that only needs a wider
    brick to overlap the layer below it), which is often possible even
    when no straight vertical filler column from ground_floating's much
    narrower strategy would ever reach the ground. Running ground_floating
    first would drop that data before repair_weak_points got a chance to
    save it."""
    best_bricks: list[Brick] | None = None
    best_score = math.inf
    for i in range(restarts):
        candidate = greedy_all(colours, catalogue, layer_type, seed=seed + i)
        candidate_graph = build_graph(candidate)
        candidate_score = score(candidate, candidate_graph, weights)
        if candidate_score < best_score:
            best_score = candidate_score
            best_bricks = candidate
    assert best_bricks is not None

    rng = np.random.default_rng(seed)
    bricks = repair_weak_points(
        best_bricks, colours, solid_occ, catalogue, layer_type, rng, max_iters=repair_iters
    )
    bricks = ground_floating(bricks, colours, solid_occ, catalogue, layer_type)

    if stability:
        bricks = repair_stability(
            bricks,
            colours,
            solid_occ,
            catalogue,
            layer_type,
            rng,
            max_iters=stability_iters,
            t_max=t_max,
            s_max=s_max,
        )

    _assert_invariants(bricks, colours)
    return bricks
