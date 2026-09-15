"""Write a small staircase model to examples/out/ for visual inspection in
BrickLink Studio: 4 layers of 2x4 bricks alternating red/blue, each stepped
one stud over from the last, with a 1x4 brick turned 90 degrees on top.

Run with:
    uv run python examples/smoke_test.py
"""

from pathlib import Path

from image2lego.colours import Colour
from image2lego.io.bricklink import write_wanted_list
from image2lego.io.ldraw import write_ldr
from image2lego.model import Brick, PartCatalogue

OUT_DIR = Path(__file__).parent / "out"

RED = 4
BLUE = 1
YELLOW = 14

# BrickLink colour ids for the base LDraw palette below. These three are
# commonly-cited stable BrickLink codes, but this example doesn't load them
# from colours.py's CSV pipeline -- double check against BrickLink's own
# colour list (https://www.bricklink.com/catalogColors.asp) before reusing
# this mapping anywhere that actually orders parts.
_BRICKLINK_COLOURS = {
    RED: Colour(ldraw_id=RED, bricklink_id=5, name="Red", rgb=(201, 26, 9)),
    BLUE: Colour(ldraw_id=BLUE, bricklink_id=7, name="Blue", rgb=(0, 85, 191)),
    YELLOW: Colour(ldraw_id=YELLOW, bricklink_id=3, name="Yellow", rgb=(245, 205, 47)),
}


def build_staircase() -> list[Brick]:
    catalogue = PartCatalogue()
    two_by_four, _ = catalogue.part(4, 2, "brick")
    one_by_four, _ = catalogue.part(1, 4, "brick")

    bricks = []
    step_colours = [RED, BLUE, RED, BLUE]
    for step, colour in enumerate(step_colours):
        bricks.append(
            Brick(x=step, y=step, z=0, w=4, d=2, colour=colour, part_id=two_by_four)
        )

    # Long axis along z (d=4 > w=1), so write_ldr emits the rotated matrix.
    bricks.append(
        Brick(x=3, y=len(step_colours), z=0, w=1, d=4, colour=YELLOW, part_id=one_by_four)
    )
    return bricks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bricks = build_staircase()

    ldr_path = OUT_DIR / "smoke.ldr"
    write_ldr(bricks, ldr_path, layer_type="brick", title="image2lego smoke test")

    xml_path = OUT_DIR / "smoke.xml"
    write_wanted_list(bricks, xml_path, colours=_BRICKLINK_COLOURS)

    print(f"Wrote {ldr_path}")
    print(f"Wrote {xml_path}")
    print()
    print("Open smoke.ldr in BrickLink Studio and check:")
    print("  [ ] All 5 bricks rest on the one below/the ground - none floating")
    print("  [ ] Studs point up")
    print("  [ ] The 4 base bricks step diagonally, alternating red/blue")
    print("  [ ] The yellow 1x4 on top is turned 90 degrees vs the 2x4s below it")
    print("      (its long side runs along z instead of x)")


if __name__ == "__main__":
    main()
