"""Simplified force-based stability analysis, after the LP formulation in
Luo, Baoxin et al. 2015, "Legolization: Optimizing LEGO Designs" -- treating
each stud joint as a small set of bounded force variables and solving a
linear program for static equilibrium of the whole assembly.

Simplifications relative to the paper (so results here aren't over-trusted
as a substitute for the real thing):

- Only static equilibrium is modelled: no dynamic/impact loads, no
  buckling of individual bricks, no material yielding beyond the linear
  force bounds below.
- Yaw (torque about the vertical y axis) is not balanced -- only force
  balance in x/y/z and torque balance about the horizontal x and z axes,
  as the task asks.
- Each stud is a single point contact at its centre, not a distributed
  pressure patch; there is no per-stud contact area or bending-moment
  capacity independent of the force bounds.
- Vertical force at a stud connection is bounded below by -T_max (tension:
  the clutch power resisting the two bricks pulling apart) and is
  otherwise unbounded above (compression, since two bricks pressed
  together can't be crushed in this model); horizontal force is bounded
  by +-S_max in both directions (shear). These are flat scalar limits, not
  a Coulomb-style friction cone whose bound depends on the normal force --
  that would make the problem non-linear (need SOCP, not LP). Ground
  contact friction reuses the same S_max bound rather than a separate
  friction coefficient.
- Ground support is a rigid, perfectly flat, compression-only contact
  (with S_max-bounded friction) under every stud of every y=0 brick.
- Mass is a simplified proportional model: footprint_area * 1.0 for a
  brick layer and footprint_area * (1/3) for a plate layer, so a brick is
  3x a plate of the same footprint and a 1x1 brick weighs exactly 1.0 --
  the same unit T_max/S_max are expressed in -- rather than the real
  physical mass of actual ABS parts.
- Only vertically-adjacent, footprint-overlapping bricks transmit force
  (build_graph's edges); bricks that merely touch side-by-side on the same
  layer are not modelled as a contact at all, since nothing in real LEGO
  interlocks them either.
- A hard equilibrium LP can be (and for a genuinely unstable structure,
  should be) infeasible, which is exactly the case we want to detect
  rather than have the solver error out on. So every equality is relaxed
  with a pair of non-negative slack variables and the LP minimises their
  sum instead. This makes the LP always solvable and turns "infeasible"
  into a continuous stress signal (total slack) rather than a solver
  failure: `feasible` is just `total_slack < eps`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse
from scipy.optimize import linprog

from image2lego.legolize.graph import build_graph
from image2lego.legolize.greedy import greedy_layer
from image2lego.legolize.repair import dissolve, expand_region_to_cover_intersecting_bricks
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

logger = logging.getLogger(__name__)

# Force-balance in x, y, z plus torque balance about x and z (yaw ignored).
_EQUATIONS = ("fx", "fy", "fz", "tx", "tz")
_N_EQUATIONS = len(_EQUATIONS)
_FX, _FY, _FZ, _TX, _TZ = range(_N_EQUATIONS)


@dataclass(frozen=True)
class ConnectionStress:
    """One stud joint between two vertically-adjacent bricks."""

    lower_brick: int
    upper_brick: int
    cell: tuple[int, int]
    force: tuple[float, float, float]
    utilisation: float


@dataclass(frozen=True)
class StabilityResult:
    feasible: bool
    total_slack: float
    brick_slack: list[float]
    connections: list[ConnectionStress]
    top_stressed_bricks: list[int]


def _layer_height(layer_type: LayerType) -> int:
    return BRICK_H_LDU if layer_type == "brick" else PLATE_H_LDU


def _brick_mass(brick: Brick, layer_type: LayerType) -> float:
    area = brick.w * brick.d
    return area * (1.0 if layer_type == "brick" else 1.0 / 3.0)


def _centroid_ldu(brick: Brick, h: int) -> tuple[float, float, float]:
    cx = (brick.x + brick.w / 2) * STUD_LDU
    cy = (brick.y + 0.5) * h
    cz = (brick.z + brick.d / 2) * STUD_LDU
    return cx, cy, cz


def analyse_stability(
    bricks: list[Brick],
    layer_type: LayerType,
    t_max: float = 1.0,
    s_max: float = 1.0,
    eps: float = 1e-6,
    top_k: int = 5,
    force_regularisation: float = 1e-4,
) -> StabilityResult:
    """Solve the relaxed static-equilibrium LP for `bricks` and report a
    per-brick slack (how far each brick is from force/torque balance),
    per-connection stud force utilisation, and the top_k most-stressed
    bricks by slack.

    force_regularisation is a small secondary objective weight on the L1
    norm of every connection/ground force. Without it, the primary
    objective (minimise total slack) is completely indifferent between any
    two force distributions that both achieve the same slack -- since
    force variables are otherwise free within their bounds, the solver can
    (and in practice reliably does) land on a vertex of the feasible
    region where forces sit pinned at +-T_max/S_max even in a fully
    feasible, lightly-loaded structure, which makes per-connection
    utilisation meaningless. This tiny tie-breaking term (kept far below
    the slack objective's weight of 1.0 so it never trades away feasible
    slack) selects the minimal-force solution among those, which is what
    makes utilisation and "most stressed brick" reflect genuine load
    concentration (e.g. at the root of a cantilever) instead of solver
    arbitrariness."""
    n = len(bricks)
    if n == 0:
        return StabilityResult(True, 0.0, [], [], [])

    h = _layer_height(layer_type)
    centroids = [_centroid_ldu(b, h) for b in bricks]
    masses = [_brick_mass(b, layer_type) for b in bricks]

    graph = build_graph(bricks)

    connections: list[tuple[int, int, tuple[int, int]]] = []
    for u, v in graph.edges:
        lower, upper = (u, v) if bricks[u].y < bricks[v].y else (v, u)
        shared = bricks[lower].footprint & bricks[upper].footprint
        connections.extend((lower, upper, cell) for cell in sorted(shared))

    ground: list[tuple[int, tuple[int, int]]] = []
    for i, brick in enumerate(bricks):
        if brick.y == 0:
            ground.extend((i, cell) for cell in sorted(brick.footprint))

    n_conn = len(connections)
    n_ground = len(ground)
    n_force_vars = 3 * n_conn + 3 * n_ground

    # Variable layout: connection (fx,fy,fz) triples, then ground (fx,fy,fz)
    # triples, then an |f| auxiliary for every one of those force
    # variables (for the regularisation objective), then a (s+, s-) pair
    # per (brick, equation).
    conn_base = 0
    ground_base = conn_base + 3 * n_conn
    reg_base = ground_base + 3 * n_ground
    slack_base = reg_base + n_force_vars
    n_vars = slack_base + 2 * n * _N_EQUATIONS

    def conn_var(idx: int, axis: int) -> int:
        return conn_base + 3 * idx + axis

    def ground_var(idx: int, axis: int) -> int:
        return ground_base + 3 * idx + axis

    def slack_var(brick_idx: int, eq_idx: int, sign: int) -> int:
        return slack_base + 2 * (brick_idx * _N_EQUATIONS + eq_idx) + sign

    def brick_row(brick_idx: int, eq_idx: int) -> int:
        return brick_idx * _N_EQUATIONS + eq_idx

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(row: int, col: int, val: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(val)

    def apply_force(
        brick_idx: int,
        var_fx: int,
        var_fy: int,
        var_fz: int,
        point: tuple[float, float, float],
        sign: float,
    ) -> None:
        cx, cy, cz = centroids[brick_idx]
        rx, ry, rz = point[0] - cx, point[1] - cy, point[2] - cz
        add(brick_row(brick_idx, _FX), var_fx, sign)
        add(brick_row(brick_idx, _FY), var_fy, sign)
        add(brick_row(brick_idx, _FZ), var_fz, sign)
        # tau = r x F: tau_x = ry*fz - rz*fy, tau_z = rx*fy - ry*fx.
        add(brick_row(brick_idx, _TX), var_fz, sign * ry)
        add(brick_row(brick_idx, _TX), var_fy, -sign * rz)
        add(brick_row(brick_idx, _TZ), var_fy, sign * rx)
        add(brick_row(brick_idx, _TZ), var_fx, -sign * ry)

    for idx, (lower, upper, (x, z)) in enumerate(connections):
        point = ((x + 0.5) * STUD_LDU, float(bricks[upper].y * h), (z + 0.5) * STUD_LDU)
        vfx, vfy, vfz = conn_var(idx, 0), conn_var(idx, 1), conn_var(idx, 2)
        apply_force(upper, vfx, vfy, vfz, point, +1.0)
        apply_force(lower, vfx, vfy, vfz, point, -1.0)

    for idx, (brick_idx, (x, z)) in enumerate(ground):
        point = ((x + 0.5) * STUD_LDU, 0.0, (z + 0.5) * STUD_LDU)
        vfx, vfy, vfz = ground_var(idx, 0), ground_var(idx, 1), ground_var(idx, 2)
        apply_force(brick_idx, vfx, vfy, vfz, point, +1.0)

    for i in range(n):
        for k in range(_N_EQUATIONS):
            row = brick_row(i, k)
            add(row, slack_var(i, k, 0), 1.0)
            add(row, slack_var(i, k, 1), -1.0)

    a_eq = scipy.sparse.coo_matrix(
        (np.array(data), (np.array(rows), np.array(cols))), shape=(n * _N_EQUATIONS, n_vars)
    ).tocsr()

    b_eq = np.zeros(n * _N_EQUATIONS)
    for i in range(n):
        # gravity: sum(F_y) - mass = 0, moved to the RHS.
        b_eq[brick_row(i, _FY)] = masses[i]

    # m >= f and m >= -f for every force variable f (its |f| auxiliary),
    # as two inequality rows each.
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_data: list[float] = []

    def add_ub(row: int, col: int, val: float) -> None:
        ub_rows.append(row)
        ub_cols.append(col)
        ub_data.append(val)

    for reg_idx, force_idx in enumerate(
        [conn_var(i, a) for i in range(n_conn) for a in range(3)]
        + [ground_var(i, a) for i in range(n_ground) for a in range(3)]
    ):
        m_idx = reg_base + reg_idx
        row = 2 * reg_idx
        add_ub(row, force_idx, 1.0)
        add_ub(row, m_idx, -1.0)
        add_ub(row + 1, force_idx, -1.0)
        add_ub(row + 1, m_idx, -1.0)

    a_ub = scipy.sparse.coo_matrix(
        (np.array(ub_data), (np.array(ub_rows), np.array(ub_cols))),
        shape=(2 * n_force_vars, n_vars),
    ).tocsr()
    b_ub = np.zeros(2 * n_force_vars)

    c = np.zeros(n_vars)
    c[reg_base:slack_base] = force_regularisation
    c[slack_base:] = 1.0

    lb = np.full(n_vars, -np.inf)
    ub = np.full(n_vars, np.inf)
    for idx in range(n_conn):
        lb[conn_var(idx, 1)] = -t_max
        lb[conn_var(idx, 0)], ub[conn_var(idx, 0)] = -s_max, s_max
        lb[conn_var(idx, 2)], ub[conn_var(idx, 2)] = -s_max, s_max
    for idx in range(n_ground):
        lb[ground_var(idx, 1)] = 0.0
        lb[ground_var(idx, 0)], ub[ground_var(idx, 0)] = -s_max, s_max
        lb[ground_var(idx, 2)], ub[ground_var(idx, 2)] = -s_max, s_max
    lb[reg_base:slack_base] = 0.0
    lb[slack_base:] = 0.0

    # scipy-stubs doesn't type A_ub/A_eq as accepting a sparse matrix, but
    # linprog genuinely does at runtime (and needs to here, for size).
    # highs-ipm (interior point), not plain "highs" (which resolves to
    # dual simplex here): a solid/symmetric structure's equilibrium LP has
    # a huge degenerate optimal face (many force distributions tie on the
    # primary slack objective, since force_regularisation's tie-break is
    # deliberately tiny), and dual simplex can take a pathological number
    # of degenerate pivots to walk it -- measured ~9x slower than IPM on a
    # 182-brick solid cube (29s vs 3s), and worse non-linearly at larger
    # sizes, for the identical optimum.
    result = linprog(
        c,
        A_eq=a_eq,  # type: ignore[call-overload]
        b_eq=b_eq,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=np.column_stack([lb, ub]),
        method="highs-ipm",
    )
    if not result.success:
        logger.warning("stability LP solver failed: %s", result.message)
        return StabilityResult(
            False, float("inf"), [float("inf")] * n, [], list(range(min(top_k, n)))
        )

    solution = result.x

    brick_slack = [
        sum(solution[slack_var(i, k, sign)] for k in range(_N_EQUATIONS) for sign in (0, 1))
        for i in range(n)
    ]
    total_slack = float(sum(brick_slack))

    conn_results: list[ConnectionStress] = []
    for idx, (lower, upper, cell) in enumerate(connections):
        fx, fy, fz = (solution[conn_var(idx, axis)] for axis in range(3))
        util_x = abs(fx) / s_max if s_max > 0 else 0.0
        util_z = abs(fz) / s_max if s_max > 0 else 0.0
        util_y = abs(fy) / t_max if fy < 0 and t_max > 0 else 0.0
        conn_results.append(
            ConnectionStress(lower, upper, cell, (fx, fy, fz), max(util_x, util_y, util_z))
        )

    # Ranking "most stressed" purely by a brick's own equilibrium slack is
    # misleading on its own: once the structure is infeasible, the LP is
    # economically indifferent to *which* brick along a load path absorbs
    # the unresolved imbalance as slack (it costs the same everywhere), so
    # slack alone can concentrate on e.g. an under-connected tip even
    # though the actual bottleneck -- the connection nearest its bound --
    # is elsewhere (typically the root of a cantilever). Each connection's
    # utilisation doesn't have that degeneracy (the regularisation term
    # breaks ties toward genuinely-needed force) and is normalised to
    # [0, 1] (it can't exceed 1: force variables are hard-bounded), so rank
    # primarily by the worst utilisation of any connection touching a
    # brick, using that brick's own slack only as a tiebreaker.
    touching_utilisation = [0.0] * n
    for conn in conn_results:
        touching_utilisation[conn.lower_brick] = max(
            touching_utilisation[conn.lower_brick], conn.utilisation
        )
        touching_utilisation[conn.upper_brick] = max(
            touching_utilisation[conn.upper_brick], conn.utilisation
        )

    order = sorted(range(n), key=lambda i: (-touching_utilisation[i], -brick_slack[i]))
    top_stressed_bricks = order[:top_k]

    return StabilityResult(
        total_slack < eps, total_slack, brick_slack, conn_results, top_stressed_bricks
    )


def repair_stability(
    bricks: list[Brick],
    colours: ColourGrid,
    solid_occ: Occupancy,
    catalogue: PartCatalogue,
    layer_type: LayerType,
    rng: np.random.Generator,
    max_iters: int = 50,
    window: tuple[int, int, int] = (2, 6, 6),
    t_max: float = 1.0,
    s_max: float = 1.0,
) -> list[Brick]:
    """Like repair.repair_weak_points, but targets the most-stressed brick
    from analyse_stability instead of graph-topology weak points, and
    accepts a dissolve/re-greedy window change only if total slack
    decreases."""
    dy, dx, dz = window
    width, height, depth = solid_occ.shape

    accepted = 0
    rejected = 0

    for _ in range(max_iters):
        result = analyse_stability(bricks, layer_type, t_max=t_max, s_max=s_max)
        if result.feasible or not result.top_stressed_bricks:
            break

        brick = bricks[result.top_stressed_bricks[0]]
        cx, cy, cz = brick.x + brick.w // 2, brick.y, brick.z + brick.d // 2

        nominal = (
            max(cy - dy // 2, 0),
            min(cy - dy // 2 + dy, height),
            max(cx - dx // 2, 0),
            min(cx - dx // 2 + dx, width),
            max(cz - dz // 2, 0),
            min(cz - dz // 2 + dz, depth),
        )
        region = expand_region_to_cover_intersecting_bricks(bricks, nominal)
        y0, y1, x0, x1, z0, z1 = region

        remaining, subgrid = dissolve(bricks, region)
        new_bricks: list[Brick] = []
        for local_y in range(subgrid.shape[1]):
            layer_bricks = greedy_layer(
                subgrid[:, local_y, :], y0 + local_y, catalogue, layer_type, rng
            )
            for b in layer_bricks:
                new_bricks.append(
                    Brick(
                        x=b.x + x0,
                        y=b.y,
                        z=b.z + z0,
                        w=b.w,
                        d=b.d,
                        colour=b.colour,
                        part_id=b.part_id,
                    )
                )

        candidate = remaining + new_bricks
        candidate_result = analyse_stability(candidate, layer_type, t_max=t_max, s_max=s_max)

        if candidate_result.total_slack < result.total_slack:
            bricks = candidate
            accepted += 1
        else:
            rejected += 1

    logger.info("repair_stability: %d accepted, %d rejected", accepted, rejected)
    return bricks
