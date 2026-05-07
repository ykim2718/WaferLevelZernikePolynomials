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


CONFIG_PATH_DEFAULT = Path.cwd() / "config.json"
SAMPLES_FOLDER_DEFAULT = Path.cwd() / "samples"
OUT_FOLDER_DEFAULT = Path.cwd() / "decomposition"

# File-name contract between generate_samples (writer) and the
# decompose / verify consumers (readers). Defined here once so all
# modules agree on the layout under samples_folder.
POINTS_JSON_FILENAME = "points_13.json"
SAMPLES_CSV_FILENAME = "samples.csv"
GROUND_TRUTH_CSV_FILENAME = "ground_truth.csv"


# -------------------------------------------------------------
# 1. Input loading - point coordinates and measurement values
# -------------------------------------------------------------
def load_wafer_coordinates(
    *,
    samples_folder: Union[str, Path],
    coordinate: CoordinateLiteral = "cartesian",
) -> pd.DataFrame:
    """Read point coordinates from points_13.json.

    Reads ONLY the fields requested by `coordinate` (no cross-field
    fallback). Returns a DataFrame indexed by point_id with the chosen
    coordinate columns.

    Parameters
    ----------
    samples_folder : Path
        Folder containing points_13.json.
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
    assert isinstance(samples_folder, (str, Path)), (
        f"samples_folder must be str/Path, got "
        f"{type(samples_folder).__name__}"
    )
    assert isinstance(coordinate, str), (
        f"coordinate must be str, got {type(coordinate).__name__}"
    )
    assert coordinate in COORDINATE_CHOICES, (
        f"coordinate must be one of {COORDINATE_CHOICES}, "
        f"got {coordinate!r}"
    )

    samples_folder = Path(samples_folder)
    with (samples_folder / POINTS_JSON_FILENAME).open() as f:
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
    samples_folder: Union[str, Path],
) -> pd.DataFrame:
    """Read samples.csv into a long-format DataFrame (T values only).

    Coordinates are NOT merged in - call `load_wafer_coordinates`
    separately and pass coords + measurements together to
    WaferLevelZernikePolynomials.

    Parameters
    ----------
    samples_folder : Path
        Folder containing samples.csv.

    Returns
    -------
    pd.DataFrame
        index   : MultiIndex(['wafer_id', 'point_id'])
        columns : ['T']  (measured thickness per wafer x point)
    """
    assert isinstance(samples_folder, (str, Path)), (
        f"samples_folder must be str/Path, got "
        f"{type(samples_folder).__name__}"
    )

    samples_folder = Path(samples_folder)
    samples_df = pd.read_csv(samples_folder / SAMPLES_CSV_FILENAME)
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
    samples_folder: Union[str, Path],
    out_folder: Union[str, Path],
    solver: SolverLiteral = "lsq",
    lam: float = 0.01,
    n_terms: int = 9,
    coordinate: CoordinateLiteral = "cartesian",
) -> List[Dict[str, Any]]:
    """Pure fitting stage: 13 measurements -> N coefficients.

    Parameters
    ----------
    samples_folder : Path holding points_13.json + samples.csv
    out_folder : where to write decomposed_samples.csv
    solver : "lsq" or "ridge"
    lam : Ridge regularization strength (used if solver="ridge")
    n_terms : number of Zernike terms
    coordinate : "cartesian" or "polar" - which fields of points_13.json
        to read (consumed by load_measured_data)

    Returns
    -------
    list of {id, coeffs (np.ndarray of n_terms)}
    """
    assert isinstance(samples_folder, (str, Path)), (
        f"samples_folder must be str/Path, got "
        f"{type(samples_folder).__name__}"
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
        samples_folder=samples_folder, coordinate=coordinate,
    )
    mesured_df = load_measured_data(samples_folder=samples_folder)

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
        description="Zernike decomposition (fitting stage)."
    )
    parser.add_argument(
        "--samples_folder", type=Path,
        default=SAMPLES_FOLDER_DEFAULT,
        help="Folder holding samples.csv + points_13.json",
    )
    parser.add_argument(
        "--out_folder", type=Path, default=OUT_FOLDER_DEFAULT,
        help="Folder to write outputs into",
    )
    parser.add_argument(
        "--config_json", type=Path, default=CONFIG_PATH_DEFAULT,
        help=f"Config JSON path (default: {CONFIG_PATH_DEFAULT.name})",
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
    with Path(args.config_json).open() as f:
        cfg = json.load(f)

    print(f"Reading from : {args.samples_folder.resolve()}")
    print(f"Writing to   : {args.out_folder.resolve()}")
    print(f"Config       : {args.config_json}")
    print(f"Solver       : {args.solver}")
    print(f"Coordinate   : {args.coordinate}")
    print()

    decompose(
        samples_folder=args.samples_folder,
        out_folder=args.out_folder,
        solver=args.solver,
        lam=args.lam,
        n_terms=cfg["decomposition"]["n_terms"],
        coordinate=args.coordinate,
    )


