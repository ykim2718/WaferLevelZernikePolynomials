"""
generate_pyramid_image.py
============================================================
Render the canonical Zernike pyramid (Noll convention) up to
radial order n_max and save it as a PNG.

This script visualizes the Zernike *basis functions themselves*
(the j -> (n, m) reference chart), independent of any wafer
measurement data. It is NOT part of the three-stage demo
pipeline -- run it whenever you need a fresh basis-function
reference image.

Outputs
-------
    zernike_pyramid.png   (default filename)

Usage
-----
    python generate_pyramid_image.py
    python generate_pyramid_image.py --n_max 5 --with_names
    python generate_pyramid_image.py --output_folder ../docs `
        --output_file pyramid_n4.png --with_names
============================================================
"""

import argparse
from pathlib import Path

from wlzpoly import ZernikePolynomials
from wlzpoly.verify import ZERNIKE_NAMES_DEFAULT


OUTPUT_FOLDER_DEFAULT = Path(__file__).parent
OUTPUT_FILE_DEFAULT = "zernike_pyramid.png"
N_MAX_DEFAULT = 4
GRID_N_DEFAULT = 200
CELL_SIZE_DEFAULT = 1.6
CMAP_DEFAULT = "RdBu_r"


# argparse help strings are kept at module level so they can fill the
# full 75-80 char line width without being squashed by argument-block
# indentation inside parse_args().

_N_MAX_HELP = (
    "Highest radial order n to draw. Row n has (n+1) cells.\n"
    "Total terms = (n_max + 1) * (n_max + 2) / 2.\n"
    "  n_max = 3 -> 10 terms\n"
    "  n_max = 4 -> 15 terms (Piston ... Spherical)\n"
    "  n_max = 5 -> 21 terms\n"
    f"(default: {N_MAX_DEFAULT})"
)

_WITH_NAMES_HELP = (
    "Add human-readable optical names (Piston, Tilt X, ...) for the "
    "(n, m) pairs listed in wlzpoly.verify.ZERNIKE_NAMES_DEFAULT.\n"
    "Cells without a configured name show only j and (n, m)."
)

_GRID_N_HELP = (
    f"Per-cell heatmap grid resolution. (default: {GRID_N_DEFAULT})"
)

_CELL_SIZE_HELP = (
    "Per-cell size in inches (cells are square). "
    f"(default: {CELL_SIZE_DEFAULT})"
)

_CMAP_HELP = (
    "Matplotlib colormap name for the heatmaps. "
    f'(default: "{CMAP_DEFAULT}")'
)

_OUTPUT_FOLDER_HELP = (
    "Folder to write the PNG into "
    "(default: Path(__file__).parent)"
)

_OUTPUT_FILE_HELP = (
    f'Filename for the PNG (default: "{OUTPUT_FILE_DEFAULT}")'
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Zernike pyramid image "
            "(basis-function reference chart, Noll convention)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--n_max", type=int, default=N_MAX_DEFAULT, help=_N_MAX_HELP,
    )
    parser.add_argument(
        "--with_names", action="store_true", help=_WITH_NAMES_HELP,
    )
    parser.add_argument(
        "--grid_n", type=int, default=GRID_N_DEFAULT,
        help=_GRID_N_HELP,
    )
    parser.add_argument(
        "--cell_size", type=float, default=CELL_SIZE_DEFAULT,
        help=_CELL_SIZE_HELP,
    )
    parser.add_argument(
        "--cmap", type=str, default=CMAP_DEFAULT, help=_CMAP_HELP,
    )
    parser.add_argument(
        "--output_folder", type=Path, default=OUTPUT_FOLDER_DEFAULT,
        help=_OUTPUT_FOLDER_HELP,
    )
    parser.add_argument(
        "--output_file", type=str, default=OUTPUT_FILE_DEFAULT,
        help=_OUTPUT_FILE_HELP,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert isinstance(args, argparse.Namespace), (
        f"args must be Namespace, got {type(args).__name__}"
    )

    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / args.output_file

    names = ZERNIKE_NAMES_DEFAULT if args.with_names else None
    n_terms = (args.n_max + 1) * (args.n_max + 2) // 2

    print("=" * 70)
    print("[generate_pyramid_image] arguments")
    print("=" * 70)
    print(f"  n_max         : {args.n_max}    (= {n_terms} terms)")
    print(f"  with_names    : {args.with_names}")
    print(f"  grid_n        : {args.grid_n}")
    print(f"  cell_size     : {args.cell_size}")
    print(f"  cmap          : {args.cmap}")
    print(f"  output path   : {out_path.resolve()}")
    print()

    png_bytes = ZernikePolynomials.pyramid_image(
        n_max=args.n_max,
        grid_n=args.grid_n,
        names=names,
        cell_size=args.cell_size,
        cmap=args.cmap,
    )
    out_path.write_bytes(png_bytes)

    print(f"-> Wrote {out_path}")
