from image2lego.legolize.graph import build_graph, grounded_components, score, weak_points
from image2lego.model import Brick


def _b(x: int, y: int, z: int, w: int, d: int, part_id: str = "3024") -> Brick:
    return Brick(x=x, y=y, z=z, w=w, d=d, colour=4, part_id=part_id)


class TestBuildGraph:
    def test_connects_overlapping_footprints_on_adjacent_layers(self) -> None:
        bricks = [_b(0, 0, 0, 2, 2), _b(0, 1, 0, 2, 2)]
        g = build_graph(bricks)
        assert g.has_edge(0, 1)

    def test_no_edge_when_footprints_dont_overlap(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1), _b(5, 1, 5, 1, 1)]
        g = build_graph(bricks)
        assert not g.has_edge(0, 1)

    def test_no_edge_when_layers_not_adjacent(self) -> None:
        bricks = [_b(0, 0, 0, 2, 2), _b(0, 2, 0, 2, 2)]
        g = build_graph(bricks)
        assert not g.has_edge(0, 1)

    def test_no_edge_within_same_layer(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1), _b(1, 0, 0, 1, 1)]
        g = build_graph(bricks)
        assert not g.has_edge(0, 1)

    def test_all_bricks_are_nodes_even_if_isolated(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1), _b(9, 9, 9, 1, 1)]
        g = build_graph(bricks)
        assert set(g.nodes) == {0, 1}


class TestGroundedComponents:
    def test_component_touching_ground_is_grounded(self) -> None:
        bricks = [_b(0, 0, 0, 2, 2), _b(0, 1, 0, 2, 2)]
        g = build_graph(bricks)
        grounded, floating = grounded_components(g, bricks)
        assert grounded == {0, 1}
        assert floating == []

    def test_component_not_touching_ground_is_floating(self) -> None:
        bricks = [_b(0, 1, 0, 2, 2), _b(0, 2, 0, 2, 2)]
        g = build_graph(bricks)
        grounded, floating = grounded_components(g, bricks)
        assert grounded == set()
        assert floating == [{0, 1}]

    def test_mixed_grounded_and_floating(self) -> None:
        bricks = [_b(0, 0, 0, 1, 1), _b(5, 3, 5, 1, 1)]
        g = build_graph(bricks)
        grounded, floating = grounded_components(g, bricks)
        assert grounded == {0}
        assert floating == [{1}]


class TestWeakPoints:
    def test_bridge_brick_is_an_articulation_point(self) -> None:
        # A single 1x1 bridging two otherwise-unconnected towers.
        bricks = [
            _b(0, 0, 0, 2, 2),
            _b(0, 1, 0, 1, 1),  # the bridge
            _b(0, 2, 0, 2, 2),
        ]
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert 1 in wp["articulation_points"]

    def test_degree_one_brick_is_single_edge(self) -> None:
        bricks = [_b(0, 0, 0, 2, 2), _b(0, 1, 0, 1, 1)]
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert 1 in wp["single_edge_bricks"]

    def test_no_weak_points_for_disconnected_single_bricks(self) -> None:
        bricks = [_b(0, 0, 0, 4, 4)]
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert wp["articulation_points"] == []
        assert wp["seams"] == []

    def test_seam_detected_across_three_consecutive_layers(self) -> None:
        # Two side-by-side 1x2 columns for 3 layers -> a straight x-boundary
        # seam at x=1 running the whole 3-layer height.
        bricks = []
        for y in range(3):
            bricks.append(_b(0, y, 0, 1, 2))
            bricks.append(_b(1, y, 0, 1, 2))
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert len(wp["seams"]) >= 1
        assert any(len(seam) >= 3 for seam in wp["seams"])

    def test_no_seam_when_boundary_only_spans_two_layers(self) -> None:
        bricks = []
        for y in range(2):
            bricks.append(_b(0, y, 0, 1, 2))
            bricks.append(_b(1, y, 0, 1, 2))
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert wp["seams"] == []

    def test_staggered_joints_produce_no_seam(self) -> None:
        # Alternate which x-boundary exists each layer -> no repeated seam.
        bricks = [
            _b(0, 0, 0, 1, 2),
            _b(1, 0, 0, 1, 2),
            _b(0, 1, 0, 2, 2),
            _b(0, 2, 0, 1, 2),
            _b(1, 2, 0, 1, 2),
        ]
        g = build_graph(bricks)
        wp = weak_points(g, bricks)
        assert wp["seams"] == []


class TestScore:
    def test_more_bricks_and_weak_points_increase_score(self) -> None:
        few = [_b(0, 0, 0, 4, 4)]
        many_isolated = [_b(x, 0, 0, 1, 1) for x in range(4)]
        g_few = build_graph(few)
        g_many = build_graph(many_isolated)
        weights = (1.0, 5.0, 2.0, 0.5)
        assert score(many_isolated, g_many, weights) > score(few, g_few, weights)

    def test_score_penalises_articulation_points(self) -> None:
        bricks = [_b(0, 0, 0, 2, 2), _b(0, 1, 0, 1, 1), _b(0, 2, 0, 2, 2)]
        g = build_graph(bricks)
        low_weight = score(bricks, g, (1.0, 0.0, 0.0, 0.0))
        high_weight = score(bricks, g, (1.0, 5.0, 0.0, 0.0))
        assert high_weight > low_weight
