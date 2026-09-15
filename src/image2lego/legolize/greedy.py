"""Greedy largest-brick-first layer merging.

For each layer, cells are scanned starting from a randomly chosen corner
(one of 4) so that the merge pattern isn't biased toward always growing
bricks in the same +x/+z direction -- since a Brick's (x, z) is always its
footprint's min corner, a candidate placement's actual origin is computed
relative to the scan direction so it always grows toward not-yet-visited
cells (see `_origin`).
"""

from __future__ import annotations

import numpy as np

from image2lego.model import Brick, ColourGrid, LayerType, PartCatalogue


def _origin(scan_coord: int, size: int, ascending: bool) -> int:
    return scan_coord if ascending else scan_coord - size + 1


def _place_largest_containing(
    colour_layer: np.ndarray,
    covered: np.ndarray,
    sx: int,
    sz: int,
    catalogue: PartCatalogue,
    layer_type: LayerType,
    rng: np.random.Generator,
    bias_cell: tuple[int, int] | None = None,
) -> Brick | None:
    """Find a placement covering cell (sx, sz). Returns None if (sx, sz) is
    already covered or every size's every offset is blocked.

    If `bias_cell` is given: prefer, above all, a placement whose footprint
    contains *both* (sx, sz) and bias_cell -- checked across every
    catalogue size from largest to smallest, using the largest size for
    which such a placement exists (ties broken by `rng`). A size that's
    merely large but doesn't happen to reach bias_cell is not enough: the
    two cells being bridged is the entire point of a biased seed, so a
    smaller placement that actually reaches it is preferred over a bigger
    one that doesn't (a first version of this function returned as soon
    as *any* size had a valid placement, tie-broken by "closest to
    bias_cell" -- but a large size's only valid placements can all fail to
    reach bias_cell while a smaller size's would have, silently producing
    a placement that satisfies seed_cell without bridging anything, which
    then never gets revisited since the caller only re-attempts the same
    seed/bias pair, not a different size). Only if no size can reach both
    cells does this fall back to the largest size with any valid
    placement, tie-broken by proximity to bias_cell.

    Without `bias_cell`, the largest valid placement wins, chosen
    uniformly among ties -- `bias_cell` matters for bridging a
    disconnection: without it, most valid offsets covering (sx, sz) don't
    happen to extend toward whatever they need to reach to actually
    restore connectivity, so a uniform choice needs many retries before
    landing on one that does."""
    width, depth = colour_layer.shape
    if covered[sx, sz]:
        return None
    colour = int(colour_layer[sx, sz])

    def _valid_offsets(fw: int, fd: int) -> list[tuple[int, int]]:
        offsets = []
        for ox in range(max(sx - fw + 1, 0), min(sx, width - fw) + 1):
            for oz in range(max(sz - fd + 1, 0), min(sz, depth - fd) + 1):
                footprint_covered = covered[ox : ox + fw, oz : oz + fd]
                if footprint_covered.any():
                    continue
                footprint_colours = colour_layer[ox : ox + fw, oz : oz + fd]
                if not np.all(footprint_colours == colour):
                    continue
                offsets.append((ox, oz))
        return offsets

    if bias_cell is not None:
        bx, bz = bias_cell
        for w, d in catalogue.sizes(layer_type):
            for fw, fd in ((w, d), (d, w)) if w != d else ((w, d),):
                bridging = [
                    (ox, oz)
                    for ox, oz in _valid_offsets(fw, fd)
                    if ox <= bx < ox + fw and oz <= bz < oz + fd
                ]
                if bridging:
                    ox, oz = bridging[int(rng.integers(0, len(bridging)))]
                    part_id, _rotated = catalogue.part(fw, fd, layer_type)
                    return Brick(x=ox, y=0, z=oz, w=fw, d=fd, colour=colour, part_id=part_id)

    for w, d in catalogue.sizes(layer_type):
        for fw, fd in ((w, d), (d, w)) if w != d else ((w, d),):
            valid_offsets = _valid_offsets(fw, fd)
            if valid_offsets:
                if bias_cell is not None:
                    bx, bz = bias_cell

                    def _distance(
                        offset: tuple[int, int],
                        fw: int = fw,
                        fd: int = fd,
                        bx: int = bx,
                        bz: int = bz,
                    ) -> int:
                        ox, oz = offset
                        dx = max(ox - bx, 0, bx - (ox + fw - 1))
                        dz = max(oz - bz, 0, bz - (oz + fd - 1))
                        return dx + dz

                    best_distance = min(_distance(o) for o in valid_offsets)
                    valid_offsets = [o for o in valid_offsets if _distance(o) == best_distance]
                ox, oz = valid_offsets[int(rng.integers(0, len(valid_offsets)))]
                part_id, _rotated = catalogue.part(fw, fd, layer_type)
                return Brick(x=ox, y=0, z=oz, w=fw, d=fd, colour=colour, part_id=part_id)

    return None


def greedy_layer(
    colour_layer: np.ndarray,
    y: int,
    catalogue: PartCatalogue,
    layer_type: LayerType,
    rng: np.random.Generator,
    seed_cell: tuple[int, int] | None = None,
    seed_bias_cell: tuple[int, int] | None = None,
) -> list[Brick]:
    """Merge one (X, Z) int16 colour layer (-1 = empty) into as few bricks
    as possible, never spanning a colour or empty-cell boundary.

    If `seed_cell` is given, the largest catalogue size that can be placed
    somewhere covering that cell is placed first (before the normal
    corner-scan pass) -- used by repair_weak_points to force a brick that
    bridges a specific disconnection point, since the plain scan's
    largest-rectangle-first bias will otherwise often strand a local
    protrusion in its own layer-below-disconnected sliver regardless of
    scan corner (there being no bias toward *any* rectangle containing a
    particular cell, just toward the largest rectangle anywhere).
    `seed_bias_cell`, if also given, additionally prefers whichever valid
    placement comes closest to it (e.g. a point on the far side of the
    disconnection being bridged), rather than picking uniformly at random
    among placements that merely happen to cover `seed_cell`."""
    width, depth = colour_layer.shape
    covered = colour_layer < 0
    bricks: list[Brick] = []

    if seed_cell is not None:
        sx, sz = seed_cell
        seed_brick = _place_largest_containing(
            colour_layer, covered, sx, sz, catalogue, layer_type, rng, bias_cell=seed_bias_cell
        )
        if seed_brick is not None:
            seed_brick = Brick(
                x=seed_brick.x,
                y=y,
                z=seed_brick.z,
                w=seed_brick.w,
                d=seed_brick.d,
                colour=seed_brick.colour,
                part_id=seed_brick.part_id,
            )
            bricks.append(seed_brick)
            covered[
                seed_brick.x : seed_brick.x + seed_brick.w,
                seed_brick.z : seed_brick.z + seed_brick.d,
            ] = True

    corner = int(rng.integers(0, 4))
    x_ascending = corner in (0, 1)
    z_ascending = corner in (0, 2)
    x_order = range(width) if x_ascending else range(width - 1, -1, -1)
    z_order = range(depth) if z_ascending else range(depth - 1, -1, -1)

    # Also vary whether (w, d) or (d, w) is tried first: otherwise every
    # layer would prefer the same split axis for non-square sizes (e.g.
    # always splitting a 4-wide area along x), producing a seam that
    # repeats on every layer regardless of the scan corner.
    prefer_wd = bool(rng.integers(0, 2))

    sizes = catalogue.sizes(layer_type)

    for x in x_order:
        for z in z_order:
            if covered[x, z]:
                continue
            colour = int(colour_layer[x, z])
            placed = False

            for w, d in sizes:
                orientations: tuple[tuple[int, int], ...]
                if w == d:
                    orientations = ((w, d),)
                elif prefer_wd:
                    orientations = ((w, d), (d, w))
                else:
                    orientations = ((d, w), (w, d))
                for fw, fd in orientations:
                    ox = _origin(x, fw, x_ascending)
                    oz = _origin(z, fd, z_ascending)
                    if ox < 0 or ox + fw > width or oz < 0 or oz + fd > depth:
                        continue

                    footprint_covered = covered[ox : ox + fw, oz : oz + fd]
                    if footprint_covered.any():
                        continue
                    footprint_colours = colour_layer[ox : ox + fw, oz : oz + fd]
                    if not np.all(footprint_colours == colour):
                        continue

                    part_id, _rotated = catalogue.part(fw, fd, layer_type)
                    bricks.append(
                        Brick(x=ox, y=y, z=oz, w=fw, d=fd, colour=colour, part_id=part_id)
                    )
                    covered[ox : ox + fw, oz : oz + fd] = True
                    placed = True
                    break
                if placed:
                    break

            if not placed:
                # Every catalogue includes a 1x1, so this should be
                # unreachable, but fall back explicitly rather than loop
                # forever if it somehow isn't.
                part_id, _rotated = catalogue.part(1, 1, layer_type)
                bricks.append(Brick(x=x, y=y, z=z, w=1, d=1, colour=colour, part_id=part_id))
                covered[x, z] = True

    return bricks


def greedy_all(
    colours: ColourGrid, catalogue: PartCatalogue, layer_type: LayerType, seed: int
) -> list[Brick]:
    """Merge every layer of a (X, Y, Z) colour grid into bricks."""
    rng = np.random.default_rng(seed)
    bricks: list[Brick] = []
    for y in range(colours.shape[1]):
        bricks.extend(greedy_layer(colours[:, y, :], y, catalogue, layer_type, rng))
    return bricks
