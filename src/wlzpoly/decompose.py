"""
decompose.py
============================================================
Zernike Decomposition (pure fitting stage)
============================================================

Read 13-point measurements and produce N Zernike coefficients per wafer.
No ground-truth comparison, no plots - that work lives in verify.py.

Inputs (under ./samples/, from generate_samples.py)
---------------------------------------------------
    points_13.json      -- 13-point coordinate definition
    samples.csv         -- id + P1..P13 measurement values

Outputs (under ./decomposition/)
--------------------------------
    decomposed_samples.csv          (id + a1..aN, fitted)
============================================================
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from .regression import SOLVER_CHOICES, SolverLiteral
from .zernike_polynomials import (
    COORDINATE_CHOICES,
    CoordinateLiteral,
    WaferLevelZernikePolynomials,
)


WORKING_FOLDER_DEFAULT = Path.cwd()
WAFER_POINTS_FILENAME_DEFAULT = "wafer_points.json"
TARGET_FILE_DEFAULT = "target_file.csv"
N_TERMS_DEFAULT = 9
OUT_FOLDER_DEFAULT = Path.cwd() / "decomposition"


def _resolve_under(p, base):
    """If `p` is absolute, return as-is; else join under `base`."""
    p = Path(p)
    base = Path(base)
    return p if p.is_absolute() else base / p


# -------------------------------------------------------------
# 1. Input loading - point coordinates and measurement values
# -------------------------------------------------------------
def load_wafer_coordinates(
    *,
    wafer_points_file: Union[str, Path],
    coordinate: CoordinateLiteral = "cartesian",
) -> pd.DataFrame:
    """Read point coordinates from a wafer-points JSON file.

    Reads ONLY the fields requested by `coordinate` (no cross-field
    fallback). Returns a DataFrame indexed by point_id with the chosen
    coordinate columns.

    Parameters
    ----------
    wafer_points_file : Path
        Path to the JSON file describing measurement points
        (commonly wafer_points.json or points_13.json).
    coordinate : "cartesian" or "polar"
        - "cartesian": read only ('x', 'y') from each point.
        - "polar":     read only ('r', 'theta') from each point.

    Returns
    -------
    pd.DataFrame
        index   : 'point_id' (str), e.g. 'P1', 'P2', ..., 'P13'
        columns :
            - if coordinate='cartesian': ['x', 'y']  (mm)
            - if coordinate='polar':     ['r', 'theta']  (mm, degrees)
        attrs['wafer_radius_mm'] : float - normalization radius from JSON.

    JSON structure
    --------------
    Cartesian fields used (when coordinate='cartesian'):
        {
          "wafer_radius_mm": 145,
          "points": [
            {"id": "P1", "x": 0,  "y": 0},
            {"id": "P2", "x": 75, "y": 0},
            ...
          ]
        }

    Polar fields used (when coordinate='polar'):
        {
          "wafer_radius_mm": 145,
          "points": [
            {"id": "P1", "r": 0,  "theta": 0},
            {"id": "P2", "r": 75, "theta": 0},
            ...
          ]
        }
    """
    assert isinstance(wafer_points_file, (str, Path)), (
        f"wafer_points_file must be str/Path, got "
        f"{type(wafer_points_file).__name__}"
    )
    assert isinstance(coordinate, str), (
        f"coordinate must be str, got {type(coordinate).__name__}"
    )
    assert coordinate in COORDINATE_CHOICES, (
        f"coordinate must be one of {COORDINATE_CHOICES}, "
        f"got {coordinate!r}"
    )

    with Path(wafer_points_file).open() as f:
        pts_def = json.load(f)
    pts = pts_def["points"]

    if coordinate == "cartesian":
        rows = [
            {"point_id": p["id"], "x": float(p["x"]), "y": float(p["y"])}
            for p in pts
        ]
        df = pd.DataFrame(rows).set_index("point_id")
    else:  # polar
        rows = [
            {
                "point_id": p["id"],
                "r": float(p["r"]),
                "theta": float(p["theta"]),
            }
            for p in pts
        ]
        df = pd.DataFrame(rows).set_index("point_id")

    df.attrs["wafer_radius_mm"] = float(pts_def["wafer_radius_mm"])
    return df


def load_measured_data(
    *,
    target_file: Union[str, Path],
) -> pd.DataFrame:
    """Read the measurement CSV into a long-format DataFrame (T only).

    Coordinates are NOT merged in - call `load_wafer_coordinates`
    separately and pass coords + measurements together to
    WaferLevelZernikePolynomials.

    Parameters
    ----------
    target_file : Path
        Path to the CSV with `id, P1, P2, ...` columns (commonly
        samples.csv or target.csv).

    Returns
    -------
    pd.DataFrame
        index   : MultiIndex(['wafer_id', 'point_id'])
        columns : ['T']  (measured thickness per wafer x point)
    """
    assert isinstance(target_file, (str, Path)), (
        f"target_file must be str/Path, got "
        f"{type(target_file).__name__}"
    )

    samples_df = pd.read_csv(Path(target_file))
    long_df = (
        samples_df
        .melt(id_vars="id", var_name="point_id", value_name="T")
        .rename(columns={"id": "wafer_id"})
        .set_index(["wafer_id", "point_id"])
        .sort_index()
    )
    return long_df


# -------------------------------------------------------------
# 2. Decompose - pure fitting
# -------------------------------------------------------------
def decompose(
    *,
    wafer_points_file: Union[str, Path],
    target_file: Union[str, Path],
    out_folder: Union[str, Path],
    solver: SolverLiteral = "lsq",
    lam: float = 0.01,
    n_terms: int = 9,
    coordinate: CoordinateLiteral = "cartesian",
) -> List[Dict[str, Any]]:
    """Pure fitting stage: N-point measurements -> N coefficients.

    Parameters
    ----------
    wafer_points_file : Path to the wafer-points JSON
    target_file : Path to the measurement CSV (id + P1..PN)
    out_folder : where to write decomposed_samples.csv
    solver : "lsq" or "ridge"
    lam : Ridge regularization strength (used if solver="ridge")
    n_terms : number of Zernike terms
    coordinate : "cartesian" or "polar" - which fields of the
        wafer-points JSON to read

    Returns
    -------
    list of {id, coeffs (np.ndarray of n_terms)}
    """
    assert isinstance(wafer_points_file, (str, Path)), (
        f"wafer_points_file must be str/Path, got "
        f"{type(wafer_points_file).__name__}"
    )
    assert isinstance(target_file, (str, Path)), (
        f"target_file must be str/Path, got "
        f"{type(target_file).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )
    assert isinstance(solver, str), (
        f"solver must be str, got {type(solver).__name__}"
    )
    assert solver in SOLVER_CHOICES, (
        f"solver must be one of {SOLVER_CHOICES}, got {solver!r}"
    )
    assert isinstance(lam, (int, float)), (
        f"lam must be number, got {type(lam).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(coordinate, str), (
        f"coordinate must be str, got {type(coordinate).__name__}"
    )
    assert coordinate in COORDINATE_CHOICES, (
        f"coordinate must be one of {COORDINATE_CHOICES}, "
        f"got {coordinate!r}"
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # ---- File I/O: load coords + measurements separately ----
    coords_df = load_wafer_coordinates(
        wafer_points_file=wafer_points_file, coordinate=coordinate,
    )
    mesured_df = load_measured_data(target_file=target_file)

    print("=" * 70)
    print(f"[decompose] solver = {solver}, n_terms = {n_terms}")
    if solver == "ridge":
        print(f"[decompose] lambda = {lam}")
    print(
        f"[decompose] fitting "
        f"{mesured_df.index.get_level_values('wafer_id').nunique()} "
        f"wafers..."
    )
    print("=" * 70)

    # ---- Pure compute: build wafer-level basis + fit per wafer ----
    wlz = WaferLevelZernikePolynomials(
        coords_df=coords_df,
        coordinate=coordinate,
        n_terms=n_terms,
    )
    fit_results = wlz.fit_coefficients(
        mesured_df=mesured_df, solver=solver, lam=lam,
    )

    # ---- File I/O: write coefficients ----
    path = out_folder / "decomposed_samples.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id"] + [f"a{j}" for j in range(1, n_terms + 1)]
        )
        for r in fit_results:
            w.writerow(
                [r["id"]] + [round(v, 4) for v in r["coeffs"]]
            )
    print(f"[decompose] -> Wrote {path}")
    return fit_results


# -------------------------------------------------------------
# 3. CLI
# -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zernike decomposition (fitting stage).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--working_folder", type=Path,
        default=WORKING_FOLDER_DEFAULT,
        help='Base folder for resolving the relative path of '
             '--wafer_points. (default: Path.cwd())',
    )
    parser.add_argument(
        "--wafer_points", type=Path,
        default=WAFER_POINTS_FILENAME_DEFAULT,
        help='Wafer measurement-point JSON. If relative, resolved '
             'under --working_folder. (default: "wafer_points.json")',
    )
    parser.add_argument(
        "--target_file", type=Path, default=TARGET_FILE_DEFAULT,
        help='Path to the measurement CSV (id + P1..PN). If relative, '
             'resolved against the current working directory. '
             '(default: "target_file.csv")',
    )
    parser.add_argument(
        "--n_terms", type=int, default=N_TERMS_DEFAULT,
        help=(
            'Number of Zernike polynomial terms (Noll j=1..n_terms) '
            'to fit.\n'
            'Names by j:\n'
            '   1  Piston       2  Tilt X       3  Tilt Y\n'
            '   4  Defocus      5  Astig 45     6  Astig 0\n'
            '   7  Coma Y       8  Coma X       9  Trefoil Y\n'
            '  10  Trefoil X   11  Spherical   ...\n'
            'Max: number of points in --wafer_points '
            '(A^T A becomes singular at the max; exceeding it is '
            'meaningless -- coefficients diverge).\n'
            'Recommended: leave >= 4 residual DOF for stable fits.\n'
            f'(default: {N_TERMS_DEFAULT})'
        ),
    )
    parser.add_argument(
        "--output_folder", type=Path, default=OUT_FOLDER_DEFAULT,
        help='Folder to write outputs into '
             '(default: Path.cwd() / "decomposition")',
    )
    parser.add_argument(
        "--solver", type=str, choices=SOLVER_CHOICES, default="lsq",
        help="Fitting solver to use (default: lsq)",
    )
    parser.add_argument(
        "--lam", type=float, default=0.01,
        help="Ridge regularization (used if --solver ridge)",
    )
    parser.add_argument(
        "--coordinate", type=str, choices=COORDINATE_CHOICES,
        default="cartesian",
        help=(
            "Which fields of points_13.json to read for the "
            "(rho, theta) grid (default: cartesian)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert isinstance(args, argparse.Namespace), (
        f"args must be Namespace, got {type(args).__name__}"
    )
    wafer_points_path = _resolve_under(
        args.wafer_points, args.working_folder,
    )

    print("=" * 70)
    print("[decompose] arguments")
    print("=" * 70)
    print(f"  Working folder : {Path(args.working_folder).resolve()}")
    print(f"  Wafer points   : {wafer_points_path}")
    print(f"  Target file    : {Path(args.target_file).resolve()}")
    print(f"  Output folder  : {args.output_folder.resolve()}")
    print(f"  n_terms        : {args.n_terms}")
    print(f"  Solver         : {args.solver}")
    print(f"  Coordinate     : {args.coordinate}")
    print()

    decompose(
        wafer_points_file=wafer_points_path,
        target_file=args.target_file,
        out_folder=args.output_folder,
        solver=args.solver,
        lam=args.lam,
        n_terms=args.n_terms,
        coordinate=args.coordinate,
    )


