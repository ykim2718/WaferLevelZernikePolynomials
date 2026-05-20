"""
reconstruct.py
============================================================
Inverse of decompose: rebuild N-point measurements from
fitted Zernike coefficients.

For each wafer, reads its a1..aN row from --input_file and
computes T = A @ a using the basis matrix derived from
--wafer_points. Writes the reconstructed measurements as a
wide CSV (one row per wafer, columns = --col_points).

R^2 is NOT computed here -- this script assumes the true
T is unknown (production / inference use case). If you DO
have a measurement-truth CSV, use a separate evaluator (e.g.
the pjt-Hexlab reconstruction_r2.py reference) downstream.

Inputs (under ./2_decomposition/, from Stage 2)
------------------------------------------------
    decomposed_targets.csv   (id + a1..aN)        --input_file
    wafer_points.json        (point layout)       --wafer_points

Outputs (under ./4_reconstruction/)
-----------------------------------
    reconstructed_targets.csv  (id + P1..PN)      --output_file
============================================================
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .decompose import _resolve_under, load_wafer_coordinates
from .zernike_polynomials import (
    COORDINATE_CHOICES,
    CoordinateLiteral,
    WaferLevelZernikePolynomials,
)


WORKING_FOLDER_DEFAULT = Path.cwd()
WAFER_POINTS_FILENAME_DEFAULT = "wafer_points.json"
INPUT_FILE_DEFAULT = "decomposed_targets.csv"
OUTPUT_FILE_DEFAULT = "reconstructed_targets.csv"
OUT_FOLDER_DEFAULT = Path.cwd() / "reconstruction"
N_TERMS_DEFAULT = 9
COL_WAFER_ID_DEFAULT = "wafer_id"
COL_POINTS_DEFAULT = [f"P{i}" for i in range(1, 14)]
COEFF_PREFIX_DEFAULT = "a"


# argparse help strings live at module level so each line can
# fill the project's 75-80 char width target (see CLAUDE.md).

_WORKING_FOLDER_HELP = (
    "Base folder for resolving the relative path of "
    "--wafer_points. (default: Path.cwd())"
)
_WAFER_POINTS_HELP = (
    "Wafer measurement-point JSON. If relative, resolved "
    'under --working_folder. (default: "wafer_points.json")'
)
_INPUT_FILE_HELP = (
    "Path to the decomposed-coefficients CSV "
    "(id + <prefix>1..<prefix>N). If relative, resolved "
    "against the current working directory. "
    f'(default: "{INPUT_FILE_DEFAULT}")'
)
_N_TERMS_HELP = (
    "Number of Zernike terms to read from --input_file "
    "(<prefix>1..<prefix>n_terms columns).\n"
    f"(default: {N_TERMS_DEFAULT})"
)
_OUTPUT_FOLDER_HELP = (
    "Folder to write the reconstructed CSV into "
    '(default: Path.cwd() / "reconstruction")'
)
_OUTPUT_FILE_HELP = (
    "Filename for the reconstructed-measurements CSV, "
    "written inside --output_folder. "
    f'(default: "{OUTPUT_FILE_DEFAULT}")'
)
_COORDINATE_HELP = (
    "Which fields of --wafer_points to read for the "
    "(rho, theta) grid. (default: cartesian)"
)
_COL_WAFER_ID_HELP = (
    "Name of the wafer-id column in --input_file (also "
    "used as the id column in the output CSV). "
    f'(default: "{COL_WAFER_ID_DEFAULT}")'
)
_COL_POINTS_HELP = (
    "Column names for the reconstructed measurement "
    "points in the output CSV. Must be a subset of the "
    "point ids in --wafer_points. "
    f"(default: {COL_POINTS_DEFAULT})"
)
_COEFF_PREFIX_HELP = (
    "Prefix for the coefficient columns in --input_file "
    "(read as <prefix>1..<prefix>n_terms). "
    f'(default: "{COEFF_PREFIX_DEFAULT}")'
)


def load_decomposed_coefficients(
    *,
    input_file: Union[str, Path],
    n_terms: int,
    col_wafer_id: str = COL_WAFER_ID_DEFAULT,
    coeff_prefix: str = COEFF_PREFIX_DEFAULT,
) -> pd.DataFrame:
    """Read decomposed-coefficients CSV (id + <prefix>1..<prefix>N).

    Returns
    -------
    pd.DataFrame
        index.name = col_wafer_id
        columns    = ['<prefix>1', '<prefix>2', ...,
                      '<prefix>n_terms']
    """
    assert isinstance(input_file, (str, Path)), (
        f"input_file must be str/Path, got "
        f"{type(input_file).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    coef_cols = [
        f"{coeff_prefix}{j}" for j in range(1, n_terms + 1)
    ]
    df = pd.read_csv(
        Path(input_file),
        usecols=[col_wafer_id] + coef_cols,
    )
    df = df.set_index(col_wafer_id)
    return df[coef_cols]


def reconstruct(
    *,
    wafer_points_file: Union[str, Path],
    input_file: Union[str, Path],
    n_terms: int = N_TERMS_DEFAULT,
    coordinate: CoordinateLiteral = "cartesian",
    col_wafer_id: str = COL_WAFER_ID_DEFAULT,
    col_points: Optional[List[str]] = None,
    coeff_prefix: str = COEFF_PREFIX_DEFAULT,
) -> pd.DataFrame:
    """Rebuild N-point measurements from decomposed coefficients.

    For each wafer row in --input_file, computes T = A @ a and
    returns a wide DataFrame with one row per wafer and one
    column per measurement point listed in `col_points`. The
    caller is responsible for persisting the result (see the
    `if __name__ == '__main__':` block for the CSV-writing
    pattern used by the CLI).

    What `T = A @ a` means
    ----------------------
    Per wafer, the reconstructed thickness at measurement point
    `i` is the inner product of basis row `i` with the
    coefficient vector::

        T[i] = a_1 * Z_1(rho_i, theta_i)
             + a_2 * Z_2(rho_i, theta_i)
             + ...
             + a_N * Z_N(rho_i, theta_i)
             = sum_k  A[i, k] * a_k

    where `A[i, k] = Z_k(rho_i, theta_i)` is the k-th Zernike
    basis evaluated at the i-th point. Stacking all N_points
    rows of A gives the matrix product `T = A @ a`. Across
    many wafers this implementation runs one matmul
    `T_recon = a_matrix @ A.T` so the output shape is
    `(n_wafers, n_points)`.

    Tiny worked example (n_terms=3, n_points=4; A values are
    illustrative, not the real Zernike values)::

        a = [500.0, 2.0, 0.5]       # piston, tilt-x, defocus
        A = [[1.0,  0.0, -1.0],     # row per measurement point
             [1.0,  1.0,  0.0],
             [1.0,  0.0,  1.0],
             [1.0, -1.0,  0.0]]
        T = A @ a
          = [1.0*500 + 0.0*2 + (-1.0)*0.5,    # = 499.5
             1.0*500 + 1.0*2 +  0.0*0.5,      # = 502.0
             1.0*500 + 0.0*2 +  1.0*0.5,      # = 500.5
             1.0*500 + (-1.0)*2 + 0.0*0.5]    # = 498.0

    Returns
    -------
    pd.DataFrame
        index.name = col_wafer_id
        columns    = col_points (in --wafer_points point_id order)
    """
    if col_points is None:
        col_points = list(COL_POINTS_DEFAULT)

    assert isinstance(wafer_points_file, (str, Path)), (
        f"wafer_points_file must be str/Path, got "
        f"{type(wafer_points_file).__name__}"
    )
    assert isinstance(input_file, (str, Path)), (
        f"input_file must be str/Path, got "
        f"{type(input_file).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert coordinate in COORDINATE_CHOICES, (
        f"coordinate must be one of {COORDINATE_CHOICES}, "
        f"got {coordinate!r}"
    )
    assert isinstance(col_wafer_id, str), (
        f"col_wafer_id must be str, got "
        f"{type(col_wafer_id).__name__}"
    )
    assert isinstance(col_points, list), (
        f"col_points must be list, got "
        f"{type(col_points).__name__}"
    )
    assert isinstance(coeff_prefix, str), (
        f"coeff_prefix must be str, got "
        f"{type(coeff_prefix).__name__}"
    )

    # ---- Build basis A (n_points x n_terms) ----
    coords_df = load_wafer_coordinates(
        wafer_points_file=wafer_points_file,
        coordinate=coordinate,
    )
    wlz = WaferLevelZernikePolynomials(
        coords_df=coords_df,
        coordinate=coordinate,
        n_terms=n_terms,
    )
    A = wlz.A
    point_order: List[str] = list(coords_df.index)

    # ---- Read fitted coefficients ----
    coef_df = load_decomposed_coefficients(
        input_file=input_file,
        n_terms=n_terms,
        col_wafer_id=col_wafer_id,
        coeff_prefix=coeff_prefix,
    )

    print("=" * 70)
    print(
        f"[reconstruct] basis A shape = {A.shape}, "
        f"n_wafers = {len(coef_df)}"
    )
    print("=" * 70)

    # ---- T_recon = a @ A^T -> (n_w, n_points) ----
    a_matrix = coef_df.to_numpy(dtype=float)
    T_recon = a_matrix @ A.T

    # Reorder output columns to match --col_points (which may be
    # a subset / permutation of the wafer_points point_id order).
    if list(col_points) != point_order:
        idx_map = {pid: i for i, pid in enumerate(point_order)}
        missing = [c for c in col_points if c not in idx_map]
        assert not missing, (
            f"--col_points {missing} not found in wafer_points "
            f"point ids {point_order}"
        )
        order_idx = [idx_map[c] for c in col_points]
        T_recon = T_recon[:, order_idx]

    return pd.DataFrame(
        T_recon,
        index=coef_df.index.rename(col_wafer_id),
        columns=col_points,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with input pre-checks."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct N-point measurements from fitted "
            "Zernike coefficients (inverse of decompose)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--working_folder", type=Path,
        default=WORKING_FOLDER_DEFAULT, help=_WORKING_FOLDER_HELP,
    )
    parser.add_argument(
        "--wafer_points", type=Path,
        default=WAFER_POINTS_FILENAME_DEFAULT,
        help=_WAFER_POINTS_HELP,
    )
    parser.add_argument(
        "--input_file", type=Path, default=INPUT_FILE_DEFAULT,
        help=_INPUT_FILE_HELP,
    )
    parser.add_argument(
        "--n_terms", type=int, default=N_TERMS_DEFAULT,
        help=_N_TERMS_HELP,
    )
    parser.add_argument(
        "--output_folder", type=Path,
        default=OUT_FOLDER_DEFAULT, help=_OUTPUT_FOLDER_HELP,
    )
    parser.add_argument(
        "--output_file", type=str, default=OUTPUT_FILE_DEFAULT,
        help=_OUTPUT_FILE_HELP,
    )
    parser.add_argument(
        "--coordinate", type=str,
        choices=COORDINATE_CHOICES, default="cartesian",
        help=_COORDINATE_HELP,
    )
    parser.add_argument(
        "--col_wafer_id", type=str,
        default=COL_WAFER_ID_DEFAULT, help=_COL_WAFER_ID_HELP,
    )
    parser.add_argument(
        "--col_points", type=str, nargs="+",
        default=list(COL_POINTS_DEFAULT), help=_COL_POINTS_HELP,
    )
    parser.add_argument(
        "--coeff_prefix", type=str,
        default=COEFF_PREFIX_DEFAULT, help=_COEFF_PREFIX_HELP,
    )
    args = parser.parse_args()

    # --- Validation: --input_file existence and column presence ---
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        parser.error(f"--input_file not found: {input_path}")
    try:
        csv_cols = pd.read_csv(
            input_path, nrows=0).columns.tolist()
    except Exception as exc:
        parser.error(
            f"--input_file is not a readable CSV "
            f"({input_path}): {exc}")
    needed = [args.col_wafer_id] + [
        f"{args.coeff_prefix}{j}"
        for j in range(1, args.n_terms + 1)
    ]
    missing = [c for c in needed if c not in csv_cols]
    if missing:
        parser.error(
            f"--input_file is missing columns {missing}. "
            f"Found: {csv_cols} ({input_path})")

    # --- Validation: --wafer_points existence + point id coverage ---
    wpts_path = _resolve_under(
        args.wafer_points, args.working_folder,
    )
    if not wpts_path.exists():
        parser.error(f"--wafer_points not found: {wpts_path}")
    try:
        with wpts_path.open() as f:
            wpts_def = json.load(f)
    except Exception as exc:
        parser.error(
            f"--wafer_points is not valid JSON "
            f"({wpts_path}): {exc}")
    json_pt_ids = [
        p.get("id") for p in wpts_def.get("points", [])
    ]
    missing_json = [
        c for c in args.col_points if c not in json_pt_ids
    ]
    if missing_json:
        parser.error(
            f"--col_points {missing_json} not found in "
            f"--wafer_points point ids {json_pt_ids} "
            f"({wpts_path})")

    return args


if __name__ == "__main__":
    args = parse_args()
    assert isinstance(args, argparse.Namespace), (
        f"args must be Namespace, got {type(args).__name__}"
    )
    wafer_points_path = _resolve_under(
        args.wafer_points, args.working_folder,
    )

    print("=" * 70)
    print("[reconstruct] arguments")
    print("=" * 70)
    print(
        f"  Working folder : "
        f"{Path(args.working_folder).resolve()}"
    )
    print(f"  Wafer points   : {wafer_points_path}")
    print(
        f"  Input file     : "
        f"{Path(args.input_file).resolve()}"
    )
    print(f"  Output folder  : {args.output_folder.resolve()}")
    print(f"  Output file    : {args.output_file}")
    print(f"  n_terms        : {args.n_terms}")
    print(f"  Coordinate     : {args.coordinate}")
    print(f"  col_wafer_id   : {args.col_wafer_id}")
    print(f"  col_points     : {args.col_points}")
    print(f"  coeff_prefix   : {args.coeff_prefix}")
    print()

    recon_df = reconstruct(
        wafer_points_file=wafer_points_path,
        input_file=args.input_file,
        n_terms=args.n_terms,
        coordinate=args.coordinate,
        col_wafer_id=args.col_wafer_id,
        col_points=args.col_points,
        coeff_prefix=args.coeff_prefix,
    )

    # ---- Write the reconstructed measurements CSV ----
    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / args.output_file
    recon_df.round(4).to_csv(out_path)
    print(f"[reconstruct] -> Wrote {out_path}")
