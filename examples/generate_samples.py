"""
generate_samples.py
============================================================
Generate synthetic Zernike-decomposed wafer thickness samples with
KNOWN ground-truth coefficients.

Configuration sources
---------------------
- config.json       : wafer geometry, scenarios, drift params,
                      decomposition settings
- points_13.json    : measurement-point coordinate definition

Outputs (under ./samples/)
--------------------------
    points_13.json       -- copy of the input, beside the data
    target_file.csv      -- id + P1..P13 (mimics metrology)
    ground_truth.csv     -- id, scenario, a1..aN (verification)
    wafer_maps.png       -- 2x3 grid of true wafer maps
    measurement_plot.png -- 13-point measurements vs true field
============================================================
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from wlzpoly import ZernikePolynomials


WORKING_FOLDER_DEFAULT = Path(__file__).parent
CONFIG_FILENAME_DEFAULT = "config.json"
WAFER_POINTS_FILENAME_DEFAULT = "wafer_points.json"
OUT_FOLDER_DEFAULT = Path(__file__).parent / "samples"

# Output filenames written by this script.  Stage 2 (decompose) reads
# TARGET_CSV_FILENAME via its --target_file flag, and Stage 3 (verify)
# reads GROUND_TRUTH_CSV_FILENAME via its --ground_truth_file flag.
# POINTS_JSON_FILENAME is a copy of the input wafer-points JSON placed
# next to the generated CSVs for downstream convenience.
POINTS_JSON_FILENAME = "points_13.json"
TARGET_CSV_FILENAME = "target_file.csv"
GROUND_TRUTH_CSV_FILENAME = "ground_truth.csv"


def _resolve_under(p, base):
    """If `p` is absolute, return as-is; else join under `base`."""
    p = Path(p)
    base = Path(base)
    return p if p.is_absolute() else base / p

# Default Gaussian noise stddev. Kept consistent across the function
# library defaults and the CLI default so callers see one value
# everywhere (set via --noise_sigma to override).
NOISE_SIGMA_DEFAULT = 5.0

# Plot styling constants
COLOR_TRUTH = "#1E2761"
COLOR_FITTED = "#F5A623"
GRID_RES = 200          # wafer-map heatmap grid resolution
DPI = 130
MAP_GRID_ROWS = 2
MAP_GRID_COLS = 3

# Noll indices that drift_series treats as "shape" terms.
# Piston (j=1) and Defocus (j=4) are handled separately (they have
# their own deterministic decay terms in cfg["drift_series"]).
DRIFT_DETERMINISTIC_INDICES = (1, 4)


def _drift_shape_indices(n_terms: int) -> Tuple[int, ...]:
    """All Noll indices in [2, n_terms] except the deterministic ones.

    These get a Gaussian random-walk perturbation in make_drift_series.
    Derived from n_terms so the drift model scales with the expansion.
    """
    return tuple(
        j for j in range(2, n_terms + 1)
        if j not in DRIFT_DETERMINISTIC_INDICES
    )


# -------------------------------------------------------------
# 1. Configuration loading
# -------------------------------------------------------------
def load_config(path: Path) -> Dict[str, Any]:
    """Load the project config JSON."""
    assert isinstance(path, (str, Path)), (
        f"path must be str/Path, got {type(path).__name__}"
    )
    with Path(path).open() as f:
        return json.load(f)


def load_points(path: Path) -> Dict[str, Any]:
    """Load the wafer measurement-point JSON."""
    assert isinstance(path, (str, Path)), (
        f"path must be str/Path, got {type(path).__name__}"
    )
    with Path(path).open() as f:
        return json.load(f)


# -------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------
def measurement_grid(
    points_def: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rho[N], theta_rad[N]) for the points.

    Reads (x, y) in mm from points_def and converts to normalized polar
    (rho = r/wafer_radius_mm, theta in radians).
    """
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    R = points_def["wafer_radius_mm"]
    pts = points_def["points"]
    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)
    rs = np.sqrt(xs ** 2 + ys ** 2) / R
    ths = np.arctan2(ys, xs)
    return rs, ths


def scenarios_with_int_keys(
    cfg: Dict[str, Any],
) -> Dict[str, Dict[int, float]]:
    """Convert scenario coefficient dicts: '1' -> 1 (int)."""
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    out: Dict[str, Dict[int, float]] = {}
    for name, coeffs in cfg["scenarios"].items():
        out[name] = {int(k): float(v) for k, v in coeffs.items()}
    return out


# -------------------------------------------------------------
# 3. Drift series
# -------------------------------------------------------------
def make_drift_series(
    cfg: Dict[str, Any],
    *,
    n_wafers: int = 30,
    n_terms: int = 9,
    seed: int = 0,
) -> List[Dict[int, float]]:
    """Build a list of wafer-coefficient dicts with drift.

    Piston (j=1) and Defocus (j=4) get deterministic decay terms,
    every other Noll index in [2, n_terms] gets a Gaussian random walk
    so the drift model scales with the chosen expansion order.
    """
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    assert isinstance(n_wafers, int), (
        f"n_wafers must be int, got {type(n_wafers).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(seed, int), (
        f"seed must be int, got {type(seed).__name__}"
    )
    rng = np.random.default_rng(seed)
    drift_p = cfg["drift_series"]
    base = scenarios_with_int_keys(cfg=cfg)["normal"]
    shape_indices = _drift_shape_indices(n_terms=n_terms)

    series: List[Dict[int, float]] = []
    for t in range(n_wafers):
        c = dict(base)
        c[1] = (
            base[1]
            + drift_p["piston_decay_per_step"] * t
            + rng.normal(0, drift_p["piston_random_sigma"])
        )
        c[4] = (
            base[4]
            + drift_p["bowl_deepen_per_step"] * t
            + rng.normal(0, drift_p["bowl_random_sigma"])
        )
        sigma = drift_p["shape_random_sigma"]
        for j in shape_indices:
            c[j] = base.get(j, 0.0) + rng.normal(0, sigma)
        series.append(c)
    return series


# -------------------------------------------------------------
# 4. Sample generation
# -------------------------------------------------------------
def generate_measurement(
    *,
    points_def: Dict[str, Any],
    coeffs: Dict[int, float],
    noise_sigma: float = NOISE_SIGMA_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one N-point measurement vector from coeffs."""
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    assert isinstance(coeffs, dict), (
        f"coeffs must be dict, got {type(coeffs).__name__}"
    )
    assert isinstance(noise_sigma, (int, float)), (
        f"noise_sigma must be number, got {type(noise_sigma).__name__}"
    )
    if rng is None:
        rng = np.random.default_rng()
    rho, theta = measurement_grid(points_def=points_def)
    z = ZernikePolynomials(coeffs=coeffs)
    T_clean = z.evaluate(rho=rho, theta=theta)
    noise = rng.normal(0.0, noise_sigma, size=T_clean.shape)
    T = T_clean + noise
    return T, T_clean


def build_dataset(
    *,
    cfg: Dict[str, Any],
    points_def: Dict[str, Any],
    n_terms: int,
    noise_sigma: float = NOISE_SIGMA_DEFAULT,
    seed: int = 42,
    n_drift: int = 30,
) -> List[Dict[str, Any]]:
    """Build a full dataset: scenarios + drift series."""
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(noise_sigma, (int, float)), (
        f"noise_sigma must be number, got {type(noise_sigma).__name__}"
    )
    assert isinstance(seed, int), (
        f"seed must be int, got {type(seed).__name__}"
    )
    assert isinstance(n_drift, int), (
        f"n_drift must be int, got {type(n_drift).__name__}"
    )

    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []
    counter = 1

    scenarios = scenarios_with_int_keys(cfg=cfg)
    coef_range = range(1, n_terms + 1)

    # ---- Scenario wafers ----
    pbar = tqdm(scenarios.items(), ncols=100, unit="wafer")
    for name, coeffs in pbar:
        wid = f"W_{counter:02d}"
        pbar.set_description(f"Generating {wid} ({name})")
        T, T_clean = generate_measurement(
            points_def=points_def,
            coeffs=coeffs,
            noise_sigma=noise_sigma,
            rng=rng,
        )
        samples.append({
            "id": wid,
            "scenario": name,
            "ground_truth": {
                f"a{j}": coeffs.get(j, 0.0) for j in coef_range
            },
            "T": T.round(4).tolist(),
            "T_clean": T_clean.round(4).tolist(),
        })
        counter += 1

    # ---- Drift series ----
    drift = make_drift_series(
        cfg=cfg, n_wafers=n_drift, n_terms=n_terms, seed=seed + 1,
    )
    pbar = tqdm(drift, ncols=100, unit="wafer")
    for coeffs in pbar:
        wid = f"W_{counter:02d}"
        pbar.set_description(f"Generating {wid} (drift)")
        T, T_clean = generate_measurement(
            points_def=points_def,
            coeffs=coeffs,
            noise_sigma=noise_sigma,
            rng=rng,
        )
        samples.append({
            "id": wid,
            "scenario": "drift",
            "ground_truth": {
                f"a{j}": round(coeffs.get(j, 0.0), 4)
                for j in coef_range
            },
            "T": T.round(4).tolist(),
            "T_clean": T_clean.round(4).tolist(),
        })
        counter += 1

    return samples


# -------------------------------------------------------------
# 5. File output
# -------------------------------------------------------------
def save_dataset(
    *,
    points_def: Dict[str, Any],
    samples: List[Dict[str, Any]],
    out_folder: Path,
    n_terms: int,
) -> None:
    """Write points_13.json copy, target_file.csv, and ground_truth.csv."""
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    assert isinstance(samples, list), (
        f"samples must be list, got {type(samples).__name__}"
    )
    assert isinstance(out_folder, (str, Path)), (
        f"out_folder must be str/Path, got {type(out_folder).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )

    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    coef_range = range(1, n_terms + 1)

    # ---- Copy points_13.json into samples/ for downstream ----
    with (out / POINTS_JSON_FILENAME).open("w") as f:
        json.dump(points_def, f, indent=2)

    # ---- Measurement-only CSV ----
    point_ids = [p["id"] for p in points_def["points"]]
    with (out / TARGET_CSV_FILENAME).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + point_ids)
        for s in samples:
            w.writerow([s["id"]] + s["T"])

    # ---- Ground-truth coefficients (separate file) ----
    with (out / GROUND_TRUTH_CSV_FILENAME).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "scenario"] + [f"a{j}" for j in coef_range]
        )
        for s in samples:
            w.writerow(
                [s["id"], s["scenario"]]
                + [s["ground_truth"][f"a{j}"] for j in coef_range]
            )


# -------------------------------------------------------------
# 6. Visualization
# -------------------------------------------------------------
def plot_scenario_maps(
    *,
    cfg: Dict[str, Any],
    points_def: Dict[str, Any],
    out_path: Path,
    grid_n: int = GRID_RES,
) -> None:
    """Render the scenario wafer maps as a 2x3 grid."""
    assert isinstance(cfg, dict), (
        f"cfg must be dict, got {type(cfg).__name__}"
    )
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    assert isinstance(out_path, (str, Path)), (
        f"out_path must be str/Path, got {type(out_path).__name__}"
    )
    assert isinstance(grid_n, int), (
        f"grid_n must be int, got {type(grid_n).__name__}"
    )

    ax_lin = np.linspace(-1, 1, grid_n)
    X, Y = np.meshgrid(ax_lin, ax_lin)
    RHO = np.sqrt(X ** 2 + Y ** 2)
    TH = np.arctan2(Y, X)
    mask = RHO <= 1.0

    fig, axes = plt.subplots(
        MAP_GRID_ROWS, MAP_GRID_COLS, figsize=(12, 8),
    )
    fig.suptitle(
        "Ground-truth wafer thickness maps (clean, no noise)",
        fontsize=13,
    )

    R = points_def["wafer_radius_mm"]
    pts = points_def["points"]
    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)
    px = xs / R
    py = ys / R

    scenarios = scenarios_with_int_keys(cfg=cfg)
    for ax, (name, coeffs) in zip(axes.ravel(), scenarios.items()):
        field = ZernikePolynomials(coeffs=coeffs).evaluate(rho=RHO, theta=TH)
        field_masked = np.where(mask, field, np.nan)
        mean_val = coeffs.get(1, 0.0)
        max_dev = float(np.nanmax(np.abs(field_masked - mean_val)))
        vmin, vmax = mean_val - max_dev, mean_val + max_dev
        im = ax.imshow(
            field_masked,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
        )
        ax.scatter(px, py, c="k", s=14, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_inspection(
    *,
    points_def: Dict[str, Any],
    samples: List[Dict[str, Any]],
    out_path: Path,
    n_terms: int,
    n_show: int = 6,
) -> None:
    """Plot T_clean vs T, plus ground-truth coefficient bars."""
    assert isinstance(points_def, dict), (
        f"points_def must be dict, got {type(points_def).__name__}"
    )
    assert isinstance(samples, list), (
        f"samples must be list, got {type(samples).__name__}"
    )
    assert isinstance(out_path, (str, Path)), (
        f"out_path must be str/Path, got {type(out_path).__name__}"
    )
    assert isinstance(n_terms, int), (
        f"n_terms must be int, got {type(n_terms).__name__}"
    )
    assert isinstance(n_show, int), (
        f"n_show must be int, got {type(n_show).__name__}"
    )

    show = [s for s in samples if not s["id"].startswith("D_")][:n_show]
    fig, axes = plt.subplots(
        len(show), 2, figsize=(11, 2.4 * len(show)),
    )
    if len(show) == 1:
        axes = np.array([axes])

    point_ids = [p["id"] for p in points_def["points"]]
    coef_range = range(1, n_terms + 1)
    shape_range = range(2, n_terms + 1)
    for i, s in enumerate(show):
        ax_t, ax_a = axes[i]

        ax_t.plot(
            point_ids, s["T_clean"], "o-",
            label="clean", color=COLOR_TRUTH,
        )
        ax_t.plot(
            point_ids, s["T"], "x",
            label="noisy", color=COLOR_FITTED,
        )
        ax_t.set_title(
            f"{s['id']}: thickness at {len(point_ids)} points",
            fontsize=10,
        )
        ax_t.tick_params(axis="x", labelsize=7, rotation=45)
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)

        coefs = [s["ground_truth"][f"a{j}"] for j in coef_range]
        bar_vals = coefs[1:]
        bar_lbls = [f"a{j}" for j in shape_range]
        colors = [
            COLOR_FITTED if abs(v) > 1.0 else COLOR_TRUTH
            for v in bar_vals
        ]
        ax_a.bar(bar_lbls, bar_vals, color=colors)
        ax_a.set_title(
            f"ground-truth a2..a{n_terms}   (a1 = {coefs[0]:.2f})",
            fontsize=10,
        )
        ax_a.axhline(0, color="k", lw=0.5)
        ax_a.tick_params(axis="x", labelsize=8)
        ax_a.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------
# 7. main
# -------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Entry point: build dataset + write all outputs."""
    assert isinstance(args, argparse.Namespace), (
        f"args must be Namespace, got {type(args).__name__}"
    )
    config_path = _resolve_under(args.config_json, args.working_folder)
    points_path = _resolve_under(args.wafer_points, args.working_folder)
    cfg = load_config(path=config_path)
    points_def = load_points(path=points_path)
    n_terms = cfg["decomposition"]["n_terms"]

    out_folder = args.output_folder
    samples = build_dataset(
        cfg=cfg,
        points_def=points_def,
        n_terms=n_terms,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        n_drift=args.n_drift,
    )
    save_dataset(
        points_def=points_def,
        samples=samples,
        out_folder=out_folder,
        n_terms=n_terms,
    )
    plot_scenario_maps(
        cfg=cfg,
        points_def=points_def,
        out_path=out_folder / "wafer_maps.png",
    )
    plot_measurement_inspection(
        points_def=points_def,
        samples=samples,
        out_path=out_folder / "measurement_plot.png",
        n_terms=n_terms,
        n_show=6,
    )

    print("=" * 70)
    print(
        f"[generate_samples] Generated {len(samples)} samples "
        f"-> {out_folder.resolve()}"
    )
    print("=" * 70)
    print(f"  working_folder = {Path(args.working_folder).resolve()}")
    print(f"  config_json    = {config_path}")
    print(f"  wafer_points   = {points_path}")
    print(f"  noise_sigma    = {args.noise_sigma}")
    print("  - points_13.json     (point definition)")
    print("  - target_file.csv    (id + P1..P13)")
    print(f"  - ground_truth.csv   (id, scenario, a1..a{n_terms})")
    print("  - wafer_maps.png     (scenario maps)")
    print("  - measurement_plot.png  (per-sample inspection)")

    print("\nScenarios (ground truth):")
    n_scenarios = len(cfg["scenarios"])
    for s in samples[:n_scenarios]:
        gt = s["ground_truth"]
        emph = max(
            (f"a{j}" for j in range(2, n_terms + 1)),
            key=lambda k: abs(gt[k]),
        )
        print(
            f"  {s['id']} ({s['scenario']:<11})  "
            f"a1={gt['a1']:>7.2f}   "
            f"dominant shape: {emph}={gt[emph]:>+6.2f}"
        )


# -------------------------------------------------------------
# 8. CLI
# -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic Zernike wafer thickness samples."
        )
    )
    parser.add_argument(
        "--noise_sigma", "-n", type=float,
        default=NOISE_SIGMA_DEFAULT,
        help=(
            f"Std-dev of Gaussian measurement noise "
            f"(default: {NOISE_SIGMA_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n_drift", type=int, default=30,
        help="Number of drift-series wafers (default: 30)",
    )
    parser.add_argument(
        "--working_folder", type=Path,
        default=WORKING_FOLDER_DEFAULT,
        help='Base folder for resolving relative paths of '
             '--config_json and --wafer_points '
             '(default: Path.cwd())',
    )
    parser.add_argument(
        "--config_json", type=Path,
        default=CONFIG_FILENAME_DEFAULT,
        help='Config JSON filename or path. If relative, resolved '
             'under --working_folder. (default: "config.json")',
    )
    parser.add_argument(
        "--wafer_points", type=Path,
        default=WAFER_POINTS_FILENAME_DEFAULT,
        help='Wafer measurement-point JSON. If relative, resolved '
             'under --working_folder. (default: "wafer_points.json")',
    )
    parser.add_argument(
        "--output_folder", type=Path, default=OUT_FOLDER_DEFAULT,
        help='Output folder '
             '(default: Path(__file__).parent / "samples")',
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(args=parse_args())
