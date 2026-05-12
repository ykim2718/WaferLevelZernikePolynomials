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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .regression import SOLVER_CHOICES, SolverLiteral
from .zernike_polynomials import ZernikePolynomials


DECOMPOSED_LSQ_FILE_DEFAULT = "decomposed_targets_lsq.csv"
DECOMPOSED_RIDGE_FILE_DEFAULT = "decomposed_targets_ridge.csv"
GROUND_TRUTH_FILE_DEFAULT = "ground_truth_file.csv"
N_TERMS_DEFAULT = 9
OUT_FOLDER_DEFAULT = Path.cwd() / "verification"

# Standard Zernike polynomial names by Noll (n, m). Used for column
# headers / chart labels when no override is supplied. Any (n, m) not
# listed here falls back to "Z(n,m)" via make_name_lookup().
ZERNIKE_NAMES_DEFAULT: Dict[tuple, str] = {
    (0, 0):  "Piston",
    (1, 1):  "Tilt X",
    (1, -1): "Tilt Y",
    (2, 0):  "Defocus",
    (2, -2): "Astig 45",
    (2, 2):  "Astig 0",
    (3, -1): "Coma Y",
    (3, 1):  "Coma X",
    (3, -3): "Trefoil Y",
    (3, 3):  "Trefoil X",
    (4, 0):  "Spherical",
}

# Plot styling constants
COLOR_TRUTH = "#1E2761"
COLOR_FITTED = "#F5A623"
DPI = 130
PISTON_YZOOM = 5.0   # +/- range around truth for Piston (a1) y-axis


# -------------------------------------------------------------
# 0. Name lookup
# -------------------------------------------------------------
def make_name_lookup(
    names_by_nm: Dict[tuple, str] = ZERNIKE_NAMES_DEFAULT,
) -> Callable[[int], str]:
    """Build a function j -> human-readable name.

    Looks up (n, m) in `names_by_nm`; falls back to 'Z(n,m)' if a
    particular pair is not configured.
    """
    assert isinstance(names_by_nm, dict), (
        f"names_by_nm must be dict, got {type(names_by_nm).__name__}"
    )

    def name(j: int) -> str:
        n, m = ZernikePolynomials.nm_from_noll(j=j)
        return names_by_nm.get((n, m), f"Z({n},{m})")
    return name


# -------------------------------------------------------------
# 1. Ground-truth loading
# -------------------------------------------------------------
def load_ground_truth(
    ground_truth_file: Union[str, Path],
    *,
    n_terms: int,
) -> Dict[str, Dict[str, Any]]:
    """Load ground_truth.csv into {id: {scenario, truth (np.ndarray)}}."""
    assert isinstance(ground_truth_file, (str, Path)), (
        f"ground_truth_file must be str/Path, got "
        f"{type(ground_truth_file).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    truth_map: Dict[str, Dict[str, Any]] = {}
    with Path(ground_truth_file).open() as f:
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
def load_decomposed_targets(
    decomposed_file: Union[str, Path],
    *,
    n_terms: int,
) -> Dict[str, np.ndarray]:
    """Load a decomposed-target CSV (id + a1..aN) into {id: coeffs}.

    Parameters
    ----------
    decomposed_file : Path
        Path to a CSV produced by `wlzpoly.decompose` with columns
        `id, a1, a2, ..., aN`.
    n_terms : int
        Number of coefficient columns to read. Must match the CSV.

    Returns
    -------
    Dict[wafer_id, np.ndarray of shape (n_terms,)]
    """
    assert isinstance(decomposed_file, (str, Path)), (
        f"decomposed_file must be str/Path, got "
        f"{type(decomposed_file).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    coeffs_by_id: Dict[str, np.ndarray] = {}
    with Path(decomposed_file).open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            coeffs_by_id[row["id"]] = np.array(
                [float(row[f"a{j}"]) for j in range(1, n_terms + 1)]
            )
    return coeffs_by_id


def verify(
    *,
    ground_truth_file: Union[str, Path],
    out_folder: Union[str, Path],
    decomposed_lsq_file: Optional[Union[str, Path]] = None,
    decomposed_ridge_file: Optional[Union[str, Path]] = None,
    n_terms: int = N_TERMS_DEFAULT,
    scenarios_to_show: List[str] = None,
    zernike_names: Dict[tuple, str] = None,
) -> List[Dict[str, Any]]:
    """Compare precomputed decomposed coefficients against ground truth.

    No fitting happens here -- both Stage 2 outputs (LSQ and/or Ridge)
    are read in directly and compared with `ground_truth_file`.

    Parameters
    ----------
    decomposed_lsq_file : optional CSV from `wlzpoly.decompose --solver lsq`.
        If None or file does not exist, LSQ is skipped.
    decomposed_ridge_file : optional CSV from `wlzpoly.decompose --solver
        ridge` (with --auto_lam or fixed --lam). If None or file does not
        exist, Ridge is skipped.
    n_terms : Zernike order (must match the decomposed CSVs).
    scenarios_to_show : scenario labels to include in the per-scenario
        table/charts. None => every unique scenario from ground_truth
        except 'drift'.
    zernike_names : (n, m) -> readable name map for labels.
        None => ZERNIKE_NAMES_DEFAULT.

    At least one of decomposed_lsq_file / decomposed_ridge_file must
    resolve to an existing file.
    """
    if zernike_names is None:
        zernike_names = ZERNIKE_NAMES_DEFAULT

    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(ground_truth_file, (str, Path)), (
        f"ground_truth_file must be str/Path, got "
        f"{type(ground_truth_file).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )

    # ---- Resolve which solvers are actually available ----
    solvers: List[SolverLiteral] = []
    coeffs_by_solver: Dict[str, Dict[str, np.ndarray]] = {}

    if decomposed_lsq_file is not None and Path(decomposed_lsq_file).is_file():
        coeffs_by_solver["lsq"] = load_decomposed_targets(
            decomposed_lsq_file, n_terms=n_terms,
        )
        solvers.append("lsq")
    if decomposed_ridge_file is not None and Path(decomposed_ridge_file).is_file():
        coeffs_by_solver["ridge"] = load_decomposed_targets(
            decomposed_ridge_file, n_terms=n_terms,
        )
        solvers.append("ridge")

    assert len(solvers) >= 1, (
        "At least one of decomposed_lsq_file or decomposed_ridge_file "
        "must exist on disk."
    )

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    name = make_name_lookup(names_by_nm=zernike_names)

    truth_map = load_ground_truth(
        ground_truth_file=ground_truth_file, n_terms=n_terms,
    )

    # ---- Determine which scenarios to show in per-scenario tables ----
    if scenarios_to_show is None:
        seen = []
        for gt in truth_map.values():
            sc = gt["scenario"]
            if sc != "drift" and sc not in seen:
                seen.append(sc)
        show_scenes = seen
    else:
        show_scenes = list(scenarios_to_show)

    print("=" * 70)
    print(f"[verify] solvers loaded = {solvers}")
    for s in solvers:
        print(
            f"[verify]   {s}: {len(coeffs_by_solver[s])} "
            f"wafer rows from decomposed CSV"
        )
    print("=" * 70)

    # ---- Build results list by joining ground_truth with each solver ----
    results: List[Dict[str, Any]] = []
    for wafer_id, gt in truth_map.items():
        truth = gt["truth"]
        scenario = gt["scenario"]
        row: Dict[str, Any] = {
            "id": wafer_id,
            "scenario": scenario,
            "truth": truth,
        }
        for s in solvers:
            coeffs = coeffs_by_solver[s].get(wafer_id)
            if coeffs is None:
                # Wafer present in truth but not in decomposed CSV.
                row[s] = np.full(n_terms, np.nan)
            else:
                row[s] = coeffs
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
        description="Verify Zernike decomposition results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--decomposed_lsq_file", type=Path,
        default=DECOMPOSED_LSQ_FILE_DEFAULT,
        help='LSQ-fitted CSV from Stage 2 (id + a1..aN). If the file '
             'does not exist, LSQ is skipped in this verify run. '
             f'(default: "{DECOMPOSED_LSQ_FILE_DEFAULT}")',
    )
    parser.add_argument(
        "--decomposed_ridge_file", type=Path,
        default=DECOMPOSED_RIDGE_FILE_DEFAULT,
        help='Ridge-fitted CSV from Stage 2 (id + a1..aN). If the file '
             'does not exist, Ridge is skipped in this verify run. '
             f'(default: "{DECOMPOSED_RIDGE_FILE_DEFAULT}")',
    )
    parser.add_argument(
        "--ground_truth_file", type=Path,
        default=GROUND_TRUTH_FILE_DEFAULT,
        help='Path to the ground-truth CSV (id, scenario, a1..aN). '
             'If relative, resolved against the current working '
             'directory. (default: "ground_truth_file.csv")',
    )
    parser.add_argument(
        "--output_folder", type=Path, default=OUT_FOLDER_DEFAULT,
        help='Folder to write outputs into '
             '(default: Path.cwd() / "verification")',
    )
    parser.add_argument(
        "--n_terms", type=int, default=N_TERMS_DEFAULT,
        help=(
            'Number of Zernike polynomial terms (Noll j=1..n_terms). '
            'Must match the Stage 2 run that produced the decomposed '
            'CSVs.\n'
            'Names by j:\n'
            '   1  Piston       2  Tilt X       3  Tilt Y\n'
            '   4  Defocus      5  Astig 45     6  Astig 0\n'
            '   7  Coma Y       8  Coma X       9  Trefoil Y\n'
            '  10  Trefoil X   11  Spherical   ...\n'
            f'(default: {N_TERMS_DEFAULT})'
        ),
    )
    parser.add_argument(
        "--scenarios_to_show", type=str, nargs="+", default=None,
        help='Scenario labels to include in the per-scenario tables '
             'and charts. Omit to use every unique scenario from '
             "ground_truth except 'drift'. (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert isinstance(args, argparse.Namespace), (
        f"args must be Namespace, got {type(args).__name__}"
    )
    lsq_path = Path(args.decomposed_lsq_file).resolve()
    ridge_path = Path(args.decomposed_ridge_file).resolve()
    ground_truth_path = Path(args.ground_truth_file).resolve()

    print("=" * 70)
    print("[verify] arguments")
    print("=" * 70)
    print(f"  Decomposed LSQ   : {lsq_path}"
          f"   {'(found)' if lsq_path.is_file() else '(missing -> skip)'}")
    print(f"  Decomposed Ridge : {ridge_path}"
          f"   {'(found)' if ridge_path.is_file() else '(missing -> skip)'}")
    print(f"  Ground truth     : {ground_truth_path}")
    print(f"  Output folder    : {args.output_folder.resolve()}")
    print(f"  n_terms          : {args.n_terms}")
    print(f"  scenarios_to_show: {args.scenarios_to_show or 'auto'}")
    print()

    verify(
        decomposed_lsq_file=lsq_path if lsq_path.is_file() else None,
        decomposed_ridge_file=ridge_path if ridge_path.is_file() else None,
        ground_truth_file=ground_truth_path,
        out_folder=args.output_folder,
        n_terms=args.n_terms,
        scenarios_to_show=args.scenarios_to_show,
    )
