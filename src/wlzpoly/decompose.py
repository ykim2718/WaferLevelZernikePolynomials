"""
decompose.py
============================================================
Zernike Decomposition (pure fitting stage)
============================================================

Read 13-point measurements and produce N Zernike coefficients per wafer.
No ground-truth comparison, no plots - that work lives in verify.py.

Inputs (under ./1_samples/, from generate_samples.py)
-----------------------------------------------------
    points_13.json      -- N-point coordinate definition (--wafer_points)
    target_file.csv     -- id + P1..PN measurement values (--input_file)

Outputs (under ./2_decomposition/)
----------------------------------
    decomposed_targets.csv          (id + a1..aN, fitted) (--output_file)
============================================================
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .regression import SOLVER_CHOICES, SolverLiteral, fit_ridge, loocv_lambda
from .zernike_polynomials import (
    COORDINATE_CHOICES,
    CoordinateLiteral,
    WaferLevelZernikePolynomials,
)


WORKING_FOLDER_DEFAULT = Path.cwd()
WAFER_POINTS_FILENAME_DEFAULT = "wafer_points.json"
INPUT_FILE_DEFAULT = "target_file.csv"
OUTPUT_FILE_DEFAULT = "decomposed_targets.csv"
N_TERMS_DEFAULT = 9
OUT_FOLDER_DEFAULT = Path.cwd() / "decomposition"

# LOOCV (used when --solver ridge --auto_lam).
LOOCV_LAMBDAS_DEFAULT = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LOOCV_REF_CHOICES = ("first_wafer", "mean", "per_wafer")
LOOCV_REF_DEFAULT = "first_wafer"


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
# 2a. Ridge + LOOCV helper
# -------------------------------------------------------------
def _decompose_ridge_with_loocv(
    *,
    wlz: WaferLevelZernikePolynomials,
    mesured_df: pd.DataFrame,
    loocv_lambdas: List[float],
    loocv_ref: str,
) -> List[Dict[str, Any]]:
    """Run Ridge fit with LOOCV-chosen lambda per `loocv_ref` strategy."""
    A = wlz.A
    point_order = list(wlz.coords_df.index)
    wafer_ids = list(
        mesured_df.index.get_level_values("wafer_id").unique()
    )

    def _t_of(wafer_id: str) -> np.ndarray:
        return (
            mesured_df.xs(wafer_id, level="wafer_id")["T"]
            .reindex(point_order)
            .to_numpy(dtype=float)
        )

    def _print_scan(best_lam: float, lam_errors: Dict[float, float],
                    label: str) -> None:
        print(f"\n[decompose] LOOCV lambda scan on {label}:")
        for lam_i, err in lam_errors.items():
            marker = "  <- best" if lam_i == best_lam else ""
            print(
                f"   lam = {lam_i:>7.3f}   "
                f"mean err^2 = {err:.4f}{marker}"
            )
        print()

    fit_results: List[Dict[str, Any]] = []

    if loocv_ref == "first_wafer":
        first_id = wafer_ids[0]
        T_ref = _t_of(first_id)
        best_lam, lam_errors = loocv_lambda(
            A=A, T=T_ref, lambdas=loocv_lambdas,
        )
        _print_scan(best_lam, lam_errors,
                    label=f"first wafer ({first_id})")
        for wid in wafer_ids:
            T = _t_of(wid)
            coeffs = fit_ridge(A=A, T=T, lam=best_lam)
            fit_results.append({"id": wid, "coeffs": coeffs})

    elif loocv_ref == "mean":
        T_mat = np.stack([_t_of(wid) for wid in wafer_ids], axis=0)
        T_ref = T_mat.mean(axis=0)
        best_lam, lam_errors = loocv_lambda(
            A=A, T=T_ref, lambdas=loocv_lambdas,
        )
        _print_scan(best_lam, lam_errors,
                    label="mean of all wafers")
        for wid in wafer_ids:
            T = _t_of(wid)
            coeffs = fit_ridge(A=A, T=T, lam=best_lam)
            fit_results.append({"id": wid, "coeffs": coeffs})

    else:  # per_wafer
        chosen_lams: List[float] = []
        for wid in wafer_ids:
            T = _t_of(wid)
            best_lam, _ = loocv_lambda(
                A=A, T=T, lambdas=loocv_lambdas,
            )
            chosen_lams.append(best_lam)
            coeffs = fit_ridge(A=A, T=T, lam=best_lam)
            fit_results.append({"id": wid, "coeffs": coeffs})
        # summary
        unique, counts = np.unique(chosen_lams, return_counts=True)
        print("\n[decompose] LOOCV per-wafer lambda distribution:")
        for lam_i, c in zip(unique, counts):
            print(
                f"   lam = {lam_i:>7.3f}   chosen by "
                f"{c:>3d} / {len(wafer_ids)} wafers"
            )
        print()

    return fit_results


# -------------------------------------------------------------
# 2. Decompose - pure fitting
# -------------------------------------------------------------
def decompose(
    *,
    wafer_points_file: Union[str, Path],
    input_file: Union[str, Path],
    out_folder: Union[str, Path],
    output_file: str = OUTPUT_FILE_DEFAULT,
    solver: SolverLiteral = "lsq",
    lam: float = 0.01,
    auto_lam: bool = False,
    loocv_lambdas: Optional[List[float]] = None,
    loocv_ref: str = LOOCV_REF_DEFAULT,
    n_terms: int = 9,
    coordinate: CoordinateLiteral = "cartesian",
) -> List[Dict[str, Any]]:
    """Pure fitting stage: N-point measurements -> N coefficients.

    Parameters
    ----------
    wafer_points_file : Path to the wafer-points JSON
    input_file : Path to the measurement CSV (id + P1..PN)
    out_folder : where to write `output_file`
    output_file : filename for the fitted-coefficients CSV
        (default: "decomposed_targets.csv")
    solver : "lsq" or "ridge"
    lam : Ridge regularization strength (used if solver="ridge"
        and auto_lam=False)
    auto_lam : when True and solver="ridge", run LOOCV to pick lambda
        from `loocv_lambdas`; `lam` is ignored
    loocv_lambdas : candidate lambdas for LOOCV (defaults to
        LOOCV_LAMBDAS_DEFAULT). Only used when auto_lam=True.
    loocv_ref : which T to feed LOOCV. One of:
        - "first_wafer" : T from the first wafer (default, fastest)
        - "mean"        : per-point mean across all wafers
        - "per_wafer"   : run LOOCV separately for every wafer
                          (per-wafer lambda)
    n_terms : number of Zernike terms
    coordinate : "cartesian" or "polar" - which fields of the
        wafer-points JSON to read

    Returns
    -------
    list of {id, coeffs (np.ndarray of n_terms)}
    """
    if loocv_lambdas is None:
        loocv_lambdas = list(LOOCV_LAMBDAS_DEFAULT)

    assert isinstance(wafer_points_file, (str, Path)), (
        f"wafer_points_file must be str/Path, got "
        f"{type(wafer_points_file).__name__}"
    )
    assert isinstance(input_file, (str, Path)), (
        f"input_file must be str/Path, got "
        f"{type(input_file).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )
    assert isinstance(output_file, str), (
        f"output_file must be str, got {type(output_file).__name__}"
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
    assert isinstance(auto_lam, bool), (
        f"auto_lam must be bool, got {type(auto_lam).__name__}"
    )
    assert isinstance(loocv_lambdas, list), (
        f"loocv_lambdas must be list, got "
        f"{type(loocv_lambdas).__name__}"
    )
    assert loocv_ref in LOOCV_REF_CHOICES, (
        f"loocv_ref must be one of {LOOCV_REF_CHOICES}, "
        f"got {loocv_ref!r}"
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
    mesured_df = load_measured_data(target_file=input_file)

    print("=" * 70)
    print(f"[decompose] solver = {solver}, n_terms = {n_terms}")
    if solver == "ridge":
        if auto_lam:
            print(
                f"[decompose] auto_lam=True, "
                f"loocv_ref={loocv_ref!r}, "
                f"loocv_lambdas={loocv_lambdas}"
            )
        else:
            print(f"[decompose] lambda = {lam} (fixed)")
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

    if solver == "ridge" and auto_lam:
        fit_results = _decompose_ridge_with_loocv(
            wlz=wlz,
            mesured_df=mesured_df,
            loocv_lambdas=loocv_lambdas,
            loocv_ref=loocv_ref,
        )
    else:
        fit_results = wlz.fit_coefficients(
            mesured_df=mesured_df, solver=solver, lam=lam,
        )

    # ---- File I/O: write coefficients ----
    path = out_folder / output_file
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
        "--input_file", type=Path, default=INPUT_FILE_DEFAULT,
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
        "--output_file", type=str, default=OUTPUT_FILE_DEFAULT,
        help='Filename for the fitted-coefficients CSV, written '
             'inside --output_folder. '
             '(default: "decomposed_targets.csv")',
    )
    parser.add_argument(
        "--solver", type=str, choices=SOLVER_CHOICES, default="lsq",
        help="Fitting solver to use (default: lsq)",
    )
    parser.add_argument(
        "--lam", type=float, default=0.01,
        help='Ridge regularization (used if --solver ridge AND '
             '--auto_lam is NOT set). (default: 0.01)',
    )
    parser.add_argument(
        "--auto_lam", action="store_true",
        help='When set together with --solver ridge, run LOOCV to '
             'pick lambda from --loocv_lambdas. --lam is ignored. '
             '(default: off)',
    )
    parser.add_argument(
        "--loocv_lambdas", type=float, nargs="+",
        default=LOOCV_LAMBDAS_DEFAULT,
        help='Candidate lambdas for LOOCV when --auto_lam is set. '
             f'(default: {LOOCV_LAMBDAS_DEFAULT})',
    )
    parser.add_argument(
        "--loocv_ref", type=str, choices=LOOCV_REF_CHOICES,
        default=LOOCV_REF_DEFAULT,
        help=(
            'Which measurement T feeds LOOCV (when --auto_lam):\n'
            '  first_wafer : T of the first wafer (fast)\n'
            '  mean        : per-point mean across all wafers\n'
            '  per_wafer   : run LOOCV per-wafer (N times slower,\n'
            '                each wafer gets its own lambda)\n'
            f'(default: {LOOCV_REF_DEFAULT})'
        ),
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
    print(f"  Input file     : {Path(args.input_file).resolve()}")
    print(f"  Output folder  : {args.output_folder.resolve()}")
    print(f"  Output file    : {args.output_file}")
    print(f"  n_terms        : {args.n_terms}")
    print(f"  Solver         : {args.solver}")
    if args.solver == "ridge":
        if args.auto_lam:
            print(f"  auto_lam       : True")
            print(f"  loocv_ref      : {args.loocv_ref}")
            print(f"  loocv_lambdas  : {args.loocv_lambdas}")
        else:
            print(f"  lam            : {args.lam}")
    print(f"  Coordinate     : {args.coordinate}")
    print()

    decompose(
        wafer_points_file=wafer_points_path,
        input_file=args.input_file,
        out_folder=args.output_folder,
        output_file=args.output_file,
        solver=args.solver,
        lam=args.lam,
        auto_lam=args.auto_lam,
        loocv_lambdas=args.loocv_lambdas,
        loocv_ref=args.loocv_ref,
        n_terms=args.n_terms,
        coordinate=args.coordinate,
    )


