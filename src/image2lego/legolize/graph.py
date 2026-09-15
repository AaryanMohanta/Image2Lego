"""Connectivity and structural-weakness analysis of a brick layout."""

from __future__ import annotations

from typing import TypedDict

import networkx as nx
import numpy as np

from image2lego.model import Brick

_MIN_SEAM_LAYERS = 3

Seam = list[tuple[int, int, int]]


class WeakPoints(TypedDict):
    articulation_points: list[int]
    single_edge_bricks: list[int]
    seams: list[Seam]


def build_graph(bricks: list[Brick]) -> nx.Graph[int]:
    """Nodes are brick indices; an edge joins two bricks on adjacent layers
    (|y1 - y2| == 1) whose footprints intersect. Built via a (y, x, z) ->
    brick-index spatial index, so this is O(total footprint cells), not
    O(n^2)."""
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(len(bricks)))

    cell_index: dict[tuple[int, int, int], int] = {}
    for i, brick in enumerate(bricks):
        for x, z in brick.footprint:
            cell_index[(brick.y, x, z)] = i

    for i, brick in enumerate(bricks):
        for x, z in brick.footprint:
            for dy in (-1, 1):
                other = cell_index.get((brick.y + dy, x, z))
                if other is not None and other != i:
                    graph.add_edge(i, other)

    return graph


def grounded_components(
    G: nx.Graph[int], bricks: list[Brick]
) -> tuple[set[int], list[set[int]]]:
    """grounded: union of all brick indices in any connected component that
    touches y=0. floating: the remaining components, each as its own set."""
    grounded: set[int] = set()
    floating: list[set[int]] = []
    for component in nx.connected_components(G):
        if any(bricks[i].y == 0 for i in component):
            grounded |= component
        else:
            floating.append(component)
    return grounded, floating


def _brick_id_grid(bricks: list[Brick]) -> np.ndarray:
    if not bricks:
        return np.full((0, 0, 0), -1, dtype=np.int64)

    max_x = max(b.x + b.w for b in bricks)
    max_y = max(b.y for b in bricks) + 1
    max_z = max(b.z + b.d for b in bricks)

    grid = np.full((max_x, max_y, max_z), -1, dtype=np.int64)
    for i, brick in enumerate(bricks):
        for x, z in brick.footprint:
            grid[x, brick.y, z] = i
    return grid


def _seams_along(grid: np.ndarray, dx: int, dz: int) -> list[Seam]:
    """Find runs of >= _MIN_SEAM_LAYERS consecutive layers where a
    brick-to-brick boundary exists at the same (x, z) -> (x+dx, z+dz)
    location."""
    width, height, depth = grid.shape
    end_x = width - dx
    end_z = depth - dz
    if end_x <= 0 or end_z <= 0:
        return []

    a = grid[:end_x, :, :end_z]
    b = grid[dx:, :, dz:]
    boundary = (a != -1) & (b != -1) & (a != b)

    seams: list[list[tuple[int, int, int]]] = []
    for x in range(end_x):
        for z in range(end_z):
            column = boundary[x, :, z]
            run_start: int | None = None
            for y in range(height + 1):
                present = y < height and bool(column[y])
                if present and run_start is None:
                    run_start = y
                elif not present and run_start is not None:
                    if y - run_start >= _MIN_SEAM_LAYERS:
                        seams.append([(x, layer, z) for layer in range(run_start, y)])
                    run_start = None
    return seams


def weak_points(G: nx.Graph[int], bricks: list[Brick]) -> WeakPoints:
    """articulation_points: cut-vertex brick indices whose removal
    disconnects their component. single_edge_bricks: brick indices with
    degree 1 (only one connection to the rest of the model). seams:
    vertical x/z-boundary joints repeated at the same location on 3+
    consecutive layers, each as a list of (x, y, z) cells."""
    articulation_points = list(nx.articulation_points(G)) if bricks else []
    single_edge_bricks = [n for n in G.nodes if G.degree[n] == 1]

    grid = _brick_id_grid(bricks)
    seams = _seams_along(grid, 1, 0) + _seams_along(grid, 0, 1)

    return {
        "articulation_points": articulation_points,
        "single_edge_bricks": single_edge_bricks,
        "seams": seams,
    }


def score(
    bricks: list[Brick], G: nx.Graph[int], weights: tuple[float, float, float, float]
) -> float:
    """cost = w0*n_bricks + w1*len(articulation) + w2*len(single_edge) +
    w3*seam_length."""
    w0, w1, w2, w3 = weights
    wp = weak_points(G, bricks)
    seam_length = sum(len(seam) for seam in wp["seams"])
    return (
        w0 * len(bricks)
        + w1 * len(wp["articulation_points"])
        + w2 * len(wp["single_edge_bricks"])
        + w3 * seam_length
    )
