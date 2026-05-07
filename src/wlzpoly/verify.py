"""
verify.py
============================================================
Zernike Decomposition - Verification stage
============================================================

Compare LSQ + Ridge fits against the ground truth, print scenario-level
tables, RMSE summary, and produce comparison plots.

Inputs (under ./samples/, from generate_samples.py)
---------------------------------------------------
    points_13.json      -- 13-point coordinate definition
    samples.csv         -- id + P1..P13 measurement values
    ground_truth.csv    -- id, scenario, a1..aN labels

Outputs (under ./verification/)
-------------------------------
    decomposition_results.csv       (truth vs fitted)
    decomposition_summary_lsq.png   (truth vs LSQ chart)
    decomposition_summary_ridge.png (truth vs Ridge chart)
============================================================
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .decompose import (
    GROUND_TRUTH_CSV_FILENAME,
    load_measured_data,
    load_wafer_coordinates,
)
from .regression import (
    SOLVER_CHOICES,
    SolverLiteral,
    fit_lsq,
    fit_ridge,
    loocv_lambda,
)
from .zernike_polynomials import (
    COORDINATE_CHOICES,
    CoordinateLiteral,
    WaferLevelZernikePolynomials,
    ZernikePolynomials,
)


CONFIG_PATH = Path.cwd() / "config.json"
SAMPLES_FOLDER_DEFAULT = Path.cwd() / "samples"
OUT_FOLDER_DEFAULT = Path.cwd() / "verification"

# Plot styling constants
COLOR_TRUTH = "#1E2761"
COLOR_FITTED = "#F5A623"
DPI = 130
PISTON_YZOOM = 5.0   # +/- range around truth for Piston (a1) y-axis


# -------------------------------------------------------------
# 0. Name lookup - config-driven
# -------------------------------------------------------------
def make_name_lookup(cfg: Dict[str, Any]) -> Callable[[int], str]:
    """Build a function j -> human-readable name.

    Looks up cfg['zernike_names'] keyed as 'n,m' strings; falls back to
    'Z(n,m)' if a particular pair is not configured.
    """
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    raw = cfg.get("zernike_names", {})
    name_by_nm: Dict[tuple, str] = {}
    for k, v in raw.items():
        n_str, m_str = k.split(",")
        name_by_nm[(int(n_str), int(m_str))] = v

    def name(j: int) -> str:
        n, m = ZernikePolynomials.nm_from_noll(j=j)
        return name_by_nm.get((n, m), f"Z({n},{m})")
    return name


# -------------------------------------------------------------
# 1. Ground-truth loading
# -------------------------------------------------------------
def load_ground_truth(
    samples_folder: Union[str, Path],
    *,
    n_terms: int,
) -> Dict[str, Dict[str, Any]]:
    """Load ground_truth.csv into {id: {scenario, truth (np.ndarray)}}."""
    assert isinstance(samples_folder, (str, Path)), (
        f"samples_folder must be str/Path, got "
        f"{type(samples_folder).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    samples_folder = Path(samples_folder)
    truth_map: Dict[str, Dict[str, Any]] = {}
    with (samples_folder / GROUND_TRUTH_CSV_FILENAME).open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            truth_map[row["id"]] = {
                "scenario": row["scenario"],
                "truth": np.array(
                    [float(row[f"a{j}"])
                     for j in range(1, n_terms + 1)]
                ),
            }
    return truth_map


# -------------------------------------------------------------
# 2. Verify - fit selected solvers, compare against truth
# -------------------------------------------------------------
def verify(
    *,
    cfg: Dict[str, Any],
    samples_folder: Union[str, Path],
    out_folder: Union[str, Path],
    solvers: List[SolverLiteral],
    coordinate: CoordinateLiteral = "cartesian",
) -> List[Dict[str, Any]]:
    """Compare selected solver fits against ground truth.

    Uses config.json for n_terms, lam candidate list, and which scenarios
    to display. Only runs solvers listed in `solvers`.

    `coordinate` selects which fields of points_13.json to read; passed
    to load_measured_data and basis_matrix_from_data.
    """
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    assert isinstance(samples_folder, (str, Path)), (
        f"samples_folder must be str/Path, got "
        f"{type(samples_folder).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )
    assert isinstance(solvers, list), (
        f"solvers must be list, got {type(solvers).__name__}"
    )
    for s in solvers:
        assert s in SOLVER_CHOICES, (
            f"solver must be one of {SOLVER_CHOICES}, got {s!r}"
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

    n_terms = cfg["decomposition"]["n_terms"]
    lambdas = cfg["decomposition"]["loocv_lambdas"]
    show_scenes = cfg["decomposition"]["scenarios_to_show"]
    name = make_name_lookup(cfg=cfg)

    coords_df = load_wafer_coordinates(
        samples_folder=samples_folder, coordinate=coordinate,
    )
    df_measured = load_measured_data(samples_folder=samples_folder)
    truth_map = load_ground_truth(
        samples_folder=samples_folder, n_terms=n_terms,
    )

    # ---- Build wafer-level ZP (precomputes basis A from coords) ----
    wlz = WaferLevelZernikePolynomials(
        coords_df=coords_df,
        coordinate=coordinate,
        n_terms=n_terms,
    )
    A = wlz.A

    print("=" * 70)
    print(
        f"[verify] basis A shape = {A.shape}, "
        f"rank = {np.linalg.matrix_rank(A)}"
    )
    print(f"[verify] cond(A^T A) = {np.linalg.cond(A.T @ A):.2e}")
    print(f"[verify] solvers = {solvers}")
    print("=" * 70)

    # ---- LOOCV lambda only if Ridge was requested ----
    best_lam = None
    # Reindex T by coords_df's point order so A's rows match T's rows.
    point_order = list(wlz.coords_df.index)

    if "ridge" in solvers:
        normal_id = next(
            sid for sid, gt in truth_map.items()
            if gt["scenario"] == "normal"
        )
        T_normal = (
            df_measured.xs(normal_id, level="wafer_id")["T"]
            .reindex(point_order)
            .to_numpy(dtype=float)
        )
        best_lam, lam_errors = loocv_lambda(
            A=A, T=T_normal, lambdas=lambdas,
        )
        print(f"\n[verify] LOOCV lam scan on '{normal_id}':")
        for lam, err in lam_errors.items():
            marker = "  <- best" if lam == best_lam else ""
            print(
                f"   lam = {lam:>7.3f}   "
                f"mean err^2 = {err:.4f}{marker}"
            )
        print()

    # ---- Fit every sample with each requested solver ----
    results: List[Dict[str, Any]] = []
    wafer_ids = (
        df_measured.index.get_level_values("wafer_id").unique()
    )
    pbar = tqdm(wafer_ids, ncols=100, unit="wafer")
    for wafer_id in pbar:
        pbar.set_description(f"Verifying {wafer_id}")
        T = (
            df_measured.xs(wafer_id, level="wafer_id")["T"]
            .reindex(point_order)
            .to_numpy(dtype=float)
        )
        gt = truth_map.get(wafer_id)
        truth = (
            gt["truth"] if gt is not None
            else np.full(n_terms, np.nan)
        )
        scenario = gt["scenario"] if gt is not None else ""
        row: Dict[str, Any] = {
            "id": wafer_id,
            "scenario": scenario,
            "truth": truth,
        }
        if "lsq" in solvers:
            row["lsq"] = fit_lsq(A=A, T=T)
        if "ridge" in solvers:
            row["ridge"] = fit_ridge(A=A, T=T, lam=best_lam)
        results.append(row)

    # ---- Scenario-level comparison (per solver) ----
    by_scenario = {
        r["scenario"]: r for r in results
        if r["scenario"] in show_scenes
    }
    for solver in solvers:
        print(f"Scenario-level comparison ({solver.upper()} fit):")
        print("-" * 70)
        header = f"{'sample':<18}" + "".join(
            f" {name(j):>9}" for j in range(1, n_terms + 1)
        )
        print(header)
        print("-" * len(header))
        for sc in show_scenes:
            r = by_scenario[sc]
            label = f"{r['id']} / {sc}"
            truth_str = f"{label:<18}" + "".join(
                f" {v:>+9.3f}" for v in r["truth"]
            )
            fit_str = f"{f'  └ {solver.upper()} fit':<18}" + "".join(
                f" {v:>+9.3f}" for v in r[solver]
            )
            err_str = f"{'  └ |error|':<18}" + "".join(
                f" {abs(d):>9.3f}"
                for d in r[solver] - r["truth"]
            )
            print(truth_str)
            print(fit_str)
            print(err_str)
            print()

    # ---- RMSE summary ----
    all_truth = np.array([r["truth"] for r in results])
    rmse_by_solver = {
        solver: np.sqrt(
            ((np.array([r[solver] for r in results]) - all_truth) ** 2)
            .mean(axis=0)
        )
        for solver in solvers
    }
    print("=" * 70)
    print(f"Per-coefficient RMSE across {len(results)} samples:")
    head = f"{'coef':<14}"
    for solver in solvers:
        head += f"{'RMSE (' + solver.upper() + ')':>16}"
    print(head)
    print("-" * len(head))
    for j in range(n_terms):
        line = f"a{j+1} ({name(j+1):<8})"
        for solver in solvers:
            line += f"{rmse_by_solver[solver][j]:>14.4f}  "
        print(line)
    if best_lam is not None:
        print(f"\nlam used for Ridge: {best_lam}")
    print("=" * 70)

    # ---- Save verification CSV + plots ----
    save_results_table(
        results=results,
        out_folder=out_folder,
        n_terms=n_terms,
        solvers=solvers,
    )
    for solver in solvers:
        out_path = out_folder / f"decomposition_summary_{solver}.png"
        plot_summary(
            results=results,
            out_path=out_path,
            show_scenes=show_scenes,
            n_terms=n_terms,
            solver=solver,
        )
        print(f"-> Wrote {out_path}")

    return results


# -------------------------------------------------------------
# 3. Verification CSV writer
# -------------------------------------------------------------
def save_results_table(
    *,
    results: List[Dict[str, Any]],
    out_folder: Union[str, Path],
    n_terms: int,
    solvers: List[SolverLiteral],
) -> None:
    """Write truth + selected solver fits per coefficient to CSV."""
    assert isinstance(results, list), (
        f"results must be list, got {type(results).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(solvers, list), (
        f"solvers must be list, got {type(solvers).__name__}"
    )

    path = Path(out_folder) / "decomposition_results.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        head = ["id", "scenario"]
        for j in range(1, n_terms + 1):
            head.append(f"a{j}_true")
            for solver in solvers:
                head.append(f"a{j}_{solver}")
        w.writerow(head)
        for r in results:
            row: List[Any] = [r["id"], r["scenario"]]
            for j in range(n_terms):
                row.append(round(r["truth"][j], 4))
                for solver in solvers:
                    row.append(round(r[solver][j], 4))
            w.writerow(row)
    print(f"-> Wrote {path}")


# -------------------------------------------------------------
# 4. Visualization
# -------------------------------------------------------------
def plot_summary(
    *,
    results: List[Dict[str, Any]],
    out_path: Union[str, Path],
    show_scenes: List[str],
    n_terms: int,
    solver: SolverLiteral = "lsq",
) -> None:
    """Show fitted vs true coefficients across scenarios."""
    assert isinstance(results, list), (
        f"results must be list, got {type(results).__name__}"
    )
    assert isinstance(out_path, (str, Path)), (
        f"out_path must be str/Path, got {type(out_path).__name__}"
    )
    assert isinstance(show_scenes, list), (
        f"show_scenes must be list, got {type(show_scenes).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(solver, str), (
        f"solver must be str, got {type(solver).__name__}"
    )
    assert solver in SOLVER_CHOICES, (
        f"solver must be one of {SOLVER_CHOICES}, got {solver!r}"
    )

    solver_label = solver.upper() if solver == "lsq" else "Ridge"

    by_scenario = {
        r["scenario"]: r for r in results
        if r["scenario"] in show_scenes
    }
    sub = [
        by_scenario[sc] for sc in show_scenes if sc in by_scenario
    ]

    fig, axes = plt.subplots(
        len(sub), 3, figsize=(13, 12),
        gridspec_kw={"width_ratios": [0.6, 1, 4]},
    )
    fig.suptitle(
        f"Recovered Zernike coefficients vs ground truth "
        f"({solver_label} fit)",
        fontsize=13,
    )

    width = 0.38
    shape_x = np.arange(2, n_terms + 1)
    for row, r in enumerate(sub):
        ax_label, ax_pist, ax_shape = axes[row]
        fit = r[solver]

        # leftmost column: id + scenario label
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.55, r["id"],
            ha="center", va="center",
            fontsize=15, fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.text(
            0.5, 0.30, r["scenario"],
            ha="center", va="center",
            fontsize=11, color="#666",
            transform=ax_label.transAxes,
        )

        # middle panel: Piston only (a1)
        ax_pist.bar(
            [-width / 2], [r["truth"][0]], width,
            label="truth", color=COLOR_TRUTH,
        )
        ax_pist.bar(
            [+width / 2], [fit[0]], width,
            label="fitted", color=COLOR_FITTED, alpha=0.85,
        )
        ax_pist.set_xticks([0])
        ax_pist.set_xticklabels(["a1 (Piston)"])
        v = r["truth"][0]
        ax_pist.set_ylim(v - PISTON_YZOOM, v + PISTON_YZOOM)
        ax_pist.set_title("mean component (a1)", fontsize=10)
        ax_pist.grid(True, alpha=0.3, axis="y")

        # right panel: a2..aN
        ax_shape.bar(
            shape_x - width / 2, r["truth"][1:], width,
            label="truth", color=COLOR_TRUTH,
        )
        ax_shape.bar(
            shape_x + width / 2, fit[1:], width,
            label="fitted", color=COLOR_FITTED, alpha=0.85,
        )
        ax_shape.axhline(0, color="k", lw=0.5)
        ax_shape.set_xticks(shape_x)
        ax_shape.set_xticklabels(
            [f"a{j}" for j in range(2, n_terms + 1)]
        )
        ax_shape.set_title(
            f"shape components (a2…a{n_terms})", fontsize=10,
        )
        ax_shape.grid(True, alpha=0.3, axis="y")
        if row == 0:
            ax_shape.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------
# 5. CLI
# -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Zernike decomposition results."
    )
    parser.add_argument(
        "--samples_folder", type=Path,
        default=SAMPLES_FOLDER_DEFAULT,
        help=(
            "Folder holding samples.csv, points_13.json, "
            "and ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--output_folder", type=Path, default=OUT_FOLDER_DEFAULT,
        help="Folder to write outputs into (default: verification)",
    )
    parser.add_argument(
        "--config_json", type=Path, default=CONFIG_PATH,
        help=f"Config JSON path (default: {CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--solver", nargs="+", choices=SOLVER_CHOICES,
        default=SOLVER_CHOICES,
        help="Which solver(s) to run (default: lsq ridge)",
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
    print(f"Writing to   : {args.output_folder.resolve()}")
    print(f"Config       : {args.config_json}")
    print(f"Solvers      : {args.solver}")
    print(f"Coordinate   : {args.coordinate}")
    print()

    verify(
        cfg=cfg,
        samples_folder=args.samples_folder,
        out_folder=args.output_folder,
        solvers=args.solver,
        coordinate=args.coordinate,
    )
