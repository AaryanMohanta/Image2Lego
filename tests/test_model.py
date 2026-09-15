import pytest

from image2lego.model import Brick, PartCatalogue


class TestBrickFootprint:
    def test_1x1_footprint_is_single_cell(self) -> None:
        brick = Brick(x=2, y=0, z=3, w=1, d=1, colour=4, part_id="3005")
        assert brick.footprint == {(2, 3)}

    def test_2x4_footprint_covers_all_cells(self) -> None:
        brick = Brick(x=0, y=0, z=0, w=2, d=4, colour=4, part_id="3001")
        assert brick.footprint == {
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
        }

    def test_footprint_offset_by_position(self) -> None:
        brick = Brick(x=5, y=1, z=7, w=1, d=2, colour=0, part_id="3004")
        assert brick.footprint == {(5, 7), (5, 8)}


class TestBrickOverlaps:
    def test_identical_footprint_same_layer_overlaps(self) -> None:
        a = Brick(x=0, y=0, z=0, w=2, d=2, colour=1, part_id="3003")
        b = Brick(x=0, y=0, z=0, w=2, d=2, colour=2, part_id="3003")
        assert a.overlaps(b) is True

    def test_partial_overlap_same_layer_overlaps(self) -> None:
        a = Brick(x=0, y=0, z=0, w=1, d=4, colour=1, part_id="3010")
        b = Brick(x=0, y=0, z=3, w=1, d=4, colour=2, part_id="3010")
        assert a.overlaps(b) is True

    def test_disjoint_footprint_same_layer_does_not_overlap(self) -> None:
        a = Brick(x=0, y=0, z=0, w=1, d=1, colour=1, part_id="3005")
        b = Brick(x=5, y=0, z=5, w=1, d=1, colour=1, part_id="3005")
        assert a.overlaps(b) is False

    def test_same_footprint_different_layer_does_not_overlap(self) -> None:
        a = Brick(x=0, y=0, z=0, w=2, d=2, colour=1, part_id="3003")
        b = Brick(x=0, y=1, z=0, w=2, d=2, colour=1, part_id="3003")
        assert a.overlaps(b) is False


class TestPartCatalogue:
    def test_1x1_brick_lookup(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(1, 1, "brick")
        assert part_id == "3005"
        assert rotated is False

    def test_1x1_plate_lookup(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(1, 1, "plate")
        assert part_id == "3024"
        assert rotated is False

    def test_2x4_brick_lookup_canonical_order(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(2, 4, "brick")
        assert part_id == "3001"
        assert rotated is False

    def test_4x2_brick_lookup_is_rotated_equivalent_of_2x4(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(4, 2, "brick")
        assert part_id == "3001"
        assert rotated is True

    def test_1x4_and_4x1_return_same_part_with_rotation_flag(self) -> None:
        catalogue = PartCatalogue()
        canonical_id, canonical_rotated = catalogue.part(1, 4, "plate")
        rotated_id, rotated_flag = catalogue.part(4, 1, "plate")
        assert canonical_id == rotated_id == "3710"
        assert canonical_rotated is False
        assert rotated_flag is True

    def test_2x6_plate_lookup(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(2, 6, "plate")
        assert part_id == "3795"
        assert rotated is False

    def test_2x8_brick_lookup(self) -> None:
        catalogue = PartCatalogue()
        part_id, rotated = catalogue.part(2, 8, "brick")
        assert part_id == "3007"
        assert rotated is False

    def test_sizes_sorted_by_area_descending(self) -> None:
        catalogue = PartCatalogue()
        sizes = catalogue.sizes("brick")
        areas = [w * d for w, d in sizes]
        assert areas == sorted(areas, reverse=True)
        assert sizes[0] == (2, 8)
        assert sizes[-1] == (1, 1)

    def test_sizes_contains_all_eleven_entries(self) -> None:
        catalogue = PartCatalogue()
        assert len(catalogue.sizes("brick")) == 11
        assert len(catalogue.sizes("plate")) == 11

    def test_name_returns_human_readable_brick_size(self) -> None:
        catalogue = PartCatalogue()
        assert catalogue.name("3001") == "Brick 2 x 4"

    def test_name_returns_human_readable_plate_size(self) -> None:
        catalogue = PartCatalogue()
        assert catalogue.name("3710") == "Plate 1 x 4"

    def test_name_raises_for_unknown_part(self) -> None:
        catalogue = PartCatalogue()
        with pytest.raises(KeyError):
            catalogue.name("9999")
