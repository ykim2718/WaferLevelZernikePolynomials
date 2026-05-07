# wlzpoly — Wafer-Level Zernike Polynomials

Decompose 13-point wafer thickness measurements into 9 Zernike coefficients (LSQ / Ridge), with a reproducible demo workflow that generates synthetic data, fits it, and verifies the recovered coefficients against ground truth.

---

## Install

```bash
pip install wlzpoly
```

Requires Python 3.9+. Dependencies: `numpy`, `pandas`, `matplotlib`, `tqdm`.

## Quick start

```python
import numpy as np
from wlzpoly import ZernikePolynomials, WaferLevelZernikePolynomials

# 1) Build a wavefront from known coefficients (Noll j → a_j)
z = ZernikePolynomials(coeffs={1: 500.0, 4: -12.0, 6: 0.5}, n_terms=9)
field = z.evaluate(rho=np.array([0.0, 0.5, 1.0]), theta=np.array([0.0, 0.0, 0.0]))

# 2) Fit Zernike coefficients from measurements at known coordinates
#    coords_df : DataFrame indexed by point_id, columns ['x','y'] (mm),
#                attrs['wafer_radius_mm']
#    df_measured : DataFrame indexed by MultiIndex(wafer_id, point_id), column ['T']
wlz = WaferLevelZernikePolynomials(
    coords_df=coords_df, coordinate="cartesian", n_terms=9,
)
fit_results = wlz.fit_coefficients(mesured_df=df_measured, solver="lsq")
# fit_results : list of {"id": <wafer_id>, "coeffs": np.ndarray}

# 3) Render a fitted wafer field
fig = wlz.draw_field(coeffs=fit_results[0]["coeffs"])
fig.savefig("W_01_fit.png", dpi=130, bbox_inches="tight")
```

`ZernikePolynomials` follows the **Noll convention** and supports any radial order (j → (n, m) is computed dynamically).

---

## Folder layout

```
WaferLevelZernikePolynomials/
│
├── pyproject.toml             ← PyPI package metadata (name="wlzpoly")
├── LICENSE                    ← MIT
├── MANIFEST.in                ← sdist inclusion rules
├── README.md
├── upload_to_pypi.ps1         ← build + twine upload
├── upload_to_github.ps1       ← idempotent git add/commit/push helper
│
├── src/
│   └── wlzpoly/               ← installed library code
│       ├── __init__.py        ← public API (ZernikePolynomials, fit_lsq, ...)
│       ├── zernike_polynomials.py  ← Zernike classes (math)
│       ├── regression.py      ← LSQ / Ridge / LOOCV solvers
│       ├── decompose.py       ← Stage 2: fitting (recover Zernike coefficients)
│       └── verify.py          ← Stage 3: verification + visualization
│
└── examples/                  ← demo (NOT installed via pip)
    ├── config.json            ← settings (scenarios, drift, decomposition)
    ├── points_13.json         ← 13-point measurement coordinates
    ├── generate_samples.py    ← Stage 1: synthetic data generation
    ├── samples/               ← Stage 1 outputs (committed for browsing)
    ├── decomposition/         ← Stage 2 outputs
    └── verification/          ← Stage 3 outputs
```

Pre-generated demo outputs are kept under `examples/{samples,decomposition,verification}/` so the figures and CSVs can be browsed directly on the GitHub page. They are excluded from the PyPI sdist via `MANIFEST.in` to keep the installed package lean.

| Output folder | Files produced |
|---|---|
| `samples/` | `points_13.json`, `samples.csv` (id + P1..P13), `ground_truth.csv` (id + scenario + a1..a9), `wafer_maps.png`, `measurement_plot.png` |
| `decomposition/` | `decomposed_samples.csv` (id + a1..a9) |
| `verification/` | `decomposition_results.csv` (truth vs lsq vs ridge), `decomposition_summary_lsq.png`, `decomposition_summary_ridge.png` |

---

## How to run the demo

### Development install (clone + editable)

```bash
git clone https://github.com/ykim2718/WaferLevelZernikePolynomials.git
cd WaferLevelZernikePolynomials
pip install -e .
```

### Three-stage demo workflow

Run from inside the `examples/` folder (the working directory anchors `config.json` and `points_13.json`).

```bash
cd examples

# Stage 1: generate synthetic measurement data
python generate_samples.py

# Stage 2: Zernike fitting -> recover 9 coefficients (LSQ or Ridge)
python -m wlzpoly.decompose --solver lsq

# Stage 3: compare against ground truth + visualize (default: LSQ + Ridge)
python -m wlzpoly.verify
python -m wlzpoly.verify --solver lsq          # LSQ only
python -m wlzpoly.verify --solver lsq ridge    # both (explicit)
```

The stages must be run in order — each one consumes the previous stage's output. Outputs land in `examples/samples/`, `examples/decomposition/`, and `examples/verification/`.

### Public API

```python
from wlzpoly import (
    ZernikePolynomials, WaferLevelZernikePolynomials, fit_lsq, fit_ridge,
)
```

---

## Data flow

```
                      ┌──────────────────┐
                      │ config.json      │  settings
                      │ points_13.json   │  coordinates
                      └────────┬─────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ generate_samples.py    │  ① data generation
                  └────────────┬───────────┘
                               │
                               ▼
                       ┌─────────────┐
                       │  samples/   │
                       └──────┬──────┘
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
    ┌──────────────────┐             ┌─────────────────┐
    │ decompose.py     │             │   verify.py     │
    │  ② fitting       │             │  ③ verify + viz │
    └─────────┬────────┘             └────────┬────────┘
              │                                │
              ▼                                ▼
      ┌──────────────┐               ┌──────────────────┐
      │decomposition/│               │  verification/   │
      └──────────────┘               └──────────────────┘
```

---

## Module reference

### `zernike_polynomials.py`

Zernike polynomial library. Follows the Noll convention; the j → (n, m) mapping is computed dynamically by the standard algorithm, so any radial order is supported.

`ZernikePolynomials` exposes **two usage modes**:

#### Mode 1: pure math (class-level)

Operations independent of any specific coefficient set. Call as `ZernikePolynomials.method()`.

| Method | Purpose |
|---|---|
| `ZernikePolynomials.nm_from_noll(j)` | Noll index j → (n, m) |
| `ZernikePolynomials.radial(n, m, rho)` | Radial polynomial R_n^m(ρ) |
| `ZernikePolynomials.basis(j, rho, theta)` | Z_j(ρ, θ) |
| `ZernikePolynomials.basis_matrix(rho, theta, n_terms=…)` | Design matrix A for fitting |
| `ZernikePolynomials.to_polar(x=…, y=…)` | Cartesian → polar (r, theta_rad) |
| `ZernikePolynomials.to_cartesian(r=…, theta=…)` | Polar → Cartesian (x, y) |
| `ZernikePolynomials.pyramid_image(n_max=…, names=…, return_type='png'\|'figure')` → `Union[bytes, Figure]` | Zernike pyramid image (PNG bytes or Figure) |

#### Mode 3: specific wafer measurement layout (subclass `WaferLevelZernikePolynomials`)

A **wafer-aware** subclass of `ZernikePolynomials`. It accepts a measurement-point coordinate frame (`coords_df`), pre-computes the basis matrix `A` once, then fits per-wafer coefficients from a measurement DataFrame (`df_measured`).

```python
from wlzpoly.decompose import load_wafer_coordinates, load_measured_data
from wlzpoly import WaferLevelZernikePolynomials

coords_df = load_wafer_coordinates(
    samples_folder="samples", coordinate="cartesian",
)
df_measured = load_measured_data(samples_folder="samples")

wlz = WaferLevelZernikePolynomials(
    coords_df=coords_df,
    coordinate="cartesian",
    n_terms=9,
)
# wlz.A : np.ndarray (m × n_terms) — pre-computed basis
fit_results = wlz.fit_coefficients(
    mesured_df=df_measured, solver="lsq",
)
# fit_results : list of {"id": <wafer_id>, "coeffs": np.ndarray}

# Render one wafer's fitted wavefront (heatmap + measurement points)
fig = wlz.draw_field(coeffs=fit_results[0]["coeffs"])
fig.savefig("W_01_fit.png", dpi=130, bbox_inches="tight")
```

**Why split it out**: the base `ZernikePolynomials` covers "coefficients are already known" scenarios (sample generation, ground truth), while `WaferLevelZernikePolynomials` covers the "coordinates + measurements → fitted coefficients" scenario. Two distinct responsibilities.

#### Mode 2: specific wavefront (instance)

An instance carrying a coefficient set — i.e. "this particular wafer/wavefront expressed as a Zernike expansion".

```python
z = ZernikePolynomials(coeffs={1: 500.0, 4: -12.0, 6: 0.5}, n_terms=9)
# Or
z = ZernikePolynomials(coeffs=[500.0, 0.5, -0.3, -2.0, 0.1, 0.2, 0, 0, 0])  # array OK

# Evaluation
field = z.evaluate(rho=rho, theta=theta)  # ndarray

# Indexing / metadata
z[4]                # -12.0 (a_4)
z[99]               # 0.0 (undefined j returns 0)
len(z)              # 9 (n_terms)
repr(z)             # "ZernikePolynomials(n_terms=9, a1=500.000, dominant=a4=-12.000)"
z.rms()             # sqrt(sum a_j^2), piston excluded by default
z.rms(exclude_piston=False)
```

For a wafer heatmap (with measurement-point overlay) use `WaferLevelZernikePolynomials.draw_field(coeffs=…)`.

**Why split it out**: `basis()` and `nm_from_noll()` are pure math that doesn't need a coefficient set, so they live at class level. `evaluate()` and `rms()` require a specific coefficient set and are therefore instance methods. Visualization (`draw_field`) only makes sense once measurement-point coordinates are known, so it lives on `WaferLevelZernikePolynomials`.

This module **does not read any external files** (no config dependency).

**Pyramid usage example**:
```python
import json
from pathlib import Path
from wlzpoly import ZernikePolynomials

# (n, m) → name mapping is sourced from config.json (optional)
cfg = json.loads(Path("config.json").read_text())
names = {
    tuple(int(x) for x in k.split(",")): v
    for k, v in cfg["zernike_names"].items()
}

# Default return_type='png' → bytes ready to write
png_bytes = ZernikePolynomials.pyramid_image(
    n_max=4,                          # n=0..4 (15 terms total)
    names=names,                      # cell labels (j, n, m only when omitted)
)
Path("zernike_pyramid.png").write_bytes(png_bytes)

# Or get the live Figure for further customization
fig = ZernikePolynomials.pyramid_image(
    n_max=4, names=names, return_type='figure',
)
fig.savefig("zernike_pyramid.png", dpi=130, bbox_inches="tight")
```

Each cell shows the Noll index j, (n, m), and (when supplied) the optical name. The sign of Z_j(ρ, θ) on the unit disk is rendered with the RdBu_r colormap. `return_type` selects PNG bytes (for saving / HTML embedding) or a live matplotlib Figure.

### `generate_samples.py` — Stage 1

Generates synthetic wafer data.

**Inputs**: `config.json`, `points_13.json`

**Outputs (`samples/`)**:
- `points_13.json` — copy of the input (consumed by later stages)
- `samples.csv` — id + P1..P13 (same shape as real metrology output)
- `ground_truth.csv` — id + scenario + a1..a9 (verification answer key)
- `wafer_maps.png` — heatmaps of all six scenarios
- `measurement_plot.png` — 13-point measurement inspection

**Procedure**:
1. Load per-scenario ground-truth coefficients (a₁..a₉) from config
2. Evaluate `T_clean = Σ a_k · Z_k(ρ_i, θ_i)` at the 13 measurement points
3. Add Gaussian noise → `T = T_clean + ε`
4. Save `T` as CSV; save the ground-truth coefficients to a separate CSV

### `regression.py`

General-purpose linear-regression solvers (no Zernike dependency).

**Provided functions**:
- `fit_lsq(A, T)` — plain least squares: â = (AᵀA)⁻¹ Aᵀ T
- `fit_ridge(A, T, lam=…)` — Ridge: â = (AᵀA + λI)⁻¹ Aᵀ T
- `loocv_lambda(A, T, lambdas=…)` — LOOCV-driven λ selection

`decompose.py` and `verify.py` import this module.

### `decompose.py` — Stage 2

Recovers 9 Zernike coefficients from the 13-point measurements (LSQ fitting).

**Inputs (`samples/`)**: `points_13.json`, `samples.csv`

**Outputs (`decomposition/`)**:
- `decomposed_samples.csv` — id + a1..a9 (production-shaped output)

**Provided functions** (also imported by `verify.py`):
- `load_wafer_coordinates(*, samples_folder, coordinate)` →
  `pd.DataFrame` (index=`point_id`, columns `x, y` or `r, theta`).
  When `--coordinate cartesian` only x and y are read; for `polar` only r and theta.
  `df.attrs['wafer_radius_mm']` is populated.
- `load_measured_data(*, samples_folder)` →
  `pd.DataFrame` (index=`MultiIndex(wafer_id, point_id)`, columns `['T']`).
  **Coordinates are not merged in** — measurements only.

**Procedure**:
1. `load_wafer_coordinates()` → coordinate DataFrame
2. `load_measured_data()` → measurement DataFrame
3. `WaferLevelZernikePolynomials(coords_df=…, coordinate=…, n_terms=…)` —
   constructor pre-computes the basis matrix `A` on `wlz.A`
4. `wlz.fit_coefficients(df_measured=…, solver=…, lam=…)` →
   per-wafer `regression.fit_lsq(A, T)` → â = (AᵀA)⁻¹ Aᵀ T.
   `T` is reindexed to `coords_df.index`, matching the row order of `A`.
5. Save the result as CSV

`ground_truth.csv` is not consumed at this stage.

### `verify.py` — Stage 3

Compares LSQ and Ridge fitting results against the ground truth.

**Inputs (`samples/`)**: `points_13.json`, `samples.csv`, `ground_truth.csv`

**Outputs (`verification/`)**:
- `decomposition_results.csv` — id + scenario + truth/lsq/ridge × 9 = 27 columns + 2 metadata
- `decomposition_summary_lsq.png` — truth vs LSQ bar chart
- `decomposition_summary_ridge.png` — truth vs Ridge bar chart

**Procedure**:
1. Fit every wafer with both LSQ and Ridge
2. Auto-select λ for Ridge via LOOCV
3. Print a per-scenario comparison table to the console
4. Print the per-coefficient RMSE summary
5. Render the six scenario wafers as PNG charts

---

## CLI options

### `generate_samples.py`

```bash
python generate_samples.py [options]
```

| Option | Default | Description |
|---|---|---|
| `--noise_sigma`, `-n` | 5.0 | Gaussian noise standard deviation |
| `--seed`, `-s` | 42 | Random seed |
| `--n_drift` | 30 | Number of wafers in the drift time series |
| `--config` | config.json | Config path |
| `--points` | points_13.json | Measurement-point definition path |
| `--out_folder` | ./samples | Output folder |

**Examples**:
```bash
python generate_samples.py --noise_sigma 0.4 --seed 100
python generate_samples.py -n 10 --n_drift 50
```

### `decompose.py`

```bash
python -m wlzpoly.decompose [options]
```

| Option | Default | Description |
|---|---|---|
| `--samples_folder` | ./samples | Input folder |
| `--out_folder` | ./decomposition | Output folder |
| `--config` | config.json | Config path |
| `--solver` | lsq | `lsq` or `ridge` |
| `--lam` | 0.01 | Ridge regularization (used when solver=ridge) |
| `--coordinate` | cartesian | `cartesian` (read x, y) or `polar` (read r, theta) |

### `verify.py`

```bash
python -m wlzpoly.verify [options]
```

| Option | Default | Description |
|---|---|---|
| `--samples_folder` | ./samples | Input folder |
| `--output_folder` | ./verification | Output folder |
| `--config_json` | config.json | Config path |
| `--solver` | lsq ridge | Solver(s) to run (`lsq`, `ridge`, or `lsq ridge`) |
| `--coordinate` | cartesian | `cartesian` (read x, y) or `polar` (read r, theta) |

### Note on the coordinate option

`points_13.json` stores both (x, y) and (r, theta) for the same 13 points. On a few diagonal edge points the stored `r=145` is rounded (the exact value is √(102.5² + 102.5²) = 144.957).

- `--coordinate cartesian` (default): read **only** (x, y) from JSON; convert to polar via the `ZernikePolynomials.to_polar(x=…, y=…)` classmethod, which uses `arctan2` and is numerically exact.
- `--coordinate polar`: read **only** (r, theta) from JSON. Theta is stored in degrees and converted with `np.deg2rad`.

The RMSE difference between the two modes is at the 4th-decimal level (e.g. a1 LSQ: cartesian 1.5461 vs polar 1.5465). The flag exists so the consumer chooses which JSON field to trust explicitly. `decompose.py`'s `load_wafer_coordinates()` / `load_measured_data()` and the `WaferLevelZernikePolynomials` class are reused by `verify.py`.

---

## Configuration files

### `config.json`

```json
{
  "wafer": {
    "size_mm": 300,
    "edge_exclusion_mm": 5,
    "fit_radius_mm": 145
  },

  "scenarios": {
    "normal":     {"1": 500.0, "2": 0.5, ..., "9": 0.0},
    "tilted":     {"1": 498.0, "2": 8.0, ...},
    "bowl":       {...},
    "astigmatic": {...},
    "trefoil":    {...},
    "comatic":    {...}
  },

  "drift_series": {
    "piston_decay_per_step": -0.15,
    "bowl_deepen_per_step":  -0.10,
    "piston_random_sigma":    0.4,
    "bowl_random_sigma":      0.3,
    "shape_random_sigma":     0.2
  },

  "decomposition": {
    "n_terms": 9,
    "loocv_lambdas": [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "scenarios_to_show": [
      "normal", "tilted", "bowl",
      "astigmatic", "trefoil", "comatic"
    ]
  },

  "zernike_names": {
    "0,0":  "Piston",
    "1,1":  "Tilt X",
    "1,-1": "Tilt Y",
    ...
  }
}
```

| Section | Meaning |
|---|---|
| `wafer` | Wafer size + edge exclusion + fitting radius |
| `scenarios` | Six ground-truth scenarios (a₁..a₉ coefficients) |
| `drift_series` | Time-series drift parameters (decay + random walk σ) |
| `decomposition` | Fitting order, LOOCV λ candidates, scenarios to display |
| `zernike_names` | (n, m) → optical standard names (Piston, Tilt X, …) |

### `points_13.json`

13-point measurement coordinate definition (cardinal-aligned pattern).

```json
{
  "wafer_size_mm": 300,
  "edge_exclusion_mm": 5,
  "wafer_radius_mm": 145,
  "pattern": "13-point cardinal-aligned",
  "points": [
    {"id": "P1",  "x": 0,    "y": 0,    "r": 0,   "theta": 0,   "zone": "Center"},
    {"id": "P2",  "x": 75,   "y": 0,    "r": 75,  "theta": 0,   "zone": "Mid_E"},
    ...
    {"id": "P13", "x": 102.5, "y": -102.5, "r": 145, "theta": 315, "zone": "Edge_SE"}
  ]
}
```

The 13-point layout:
- **P1**: Center (1)
- **P2..P5**: Middle ring r=75mm at 0°/90°/180°/270° (4)
- **P6..P13**: Edge ring r=145mm at 0°/45°/.../315° (8)

---

## Output files explained

### `samples.csv` (Stage 1 output)

```
id,P1,P2,P3,...,P13
W_01,504.99,497.06,504.92,...,502.55
W_02,507.97,510.82,493.97,...,506.47
...
```

13-point thickness measurements — same shape as a real metrology export.

### `ground_truth.csv` (Stage 1 output)

```
id,scenario,a1,a2,a3,...,a9
W_01,normal,500.0,0.5,-0.3,...,0.0
W_02,tilted,498.0,8.0,-1.5,...,0.0
```

Per-wafer ground-truth Zernike coefficients (used only for verification).

### `decomposed_samples.csv` (Stage 2 output)

```
id,a1,a2,a3,...,a9
W_01,499.27,-1.99,2.16,...,0.21
W_02,498.50,8.05,-1.61,...,-0.05
```

LSQ-recovered coefficients — the 13 → 9 compression result (production output).

### `decomposition_results.csv` (Stage 3 output)

```
id,scenario,a1_true,a1_lsq,a1_ridge,a2_true,a2_lsq,a2_ridge,...
W_01,normal,500.0,499.27,499.20,0.5,-1.99,-1.95,...
```

For every coefficient, three columns: truth / LSQ / Ridge → 27 columns + 2 metadata.

### `decomposition_summary_*.png` (Stage 3 output)

Six scenario wafers, one row each. Every row has three panels:

```
┌──────────────┬───────────────┬─────────────────────────────────┐
│  W_01        │   mean (a1)   │   shape components (a2..a9)    │
│  normal      │   [narrow bar]│   [eight wide bars]             │
└──────────────┴───────────────┴─────────────────────────────────┘
```

- **Left**: id + scenario label
- **Middle**: a₁ (Piston, mean thickness) — y-axis zoomed to ±5
- **Right**: a₂..a₉ (shape components)

Every bar panel pairs navy (truth) with orange (fitted). The closer the bars overlap, the more accurate the fit.

### Console output sample (`verify.py`)

```
Per-coefficient RMSE across 36 samples:
coef          RMSE (LSQ)    RMSE (Ridge)
----------------------------------------
a1 (Piston  )    1.5464        1.6408
a2 (Tilt X  )    1.6980        1.6955
...
a9 (Trefoil Y)   0.8764        0.8762

λ used for Ridge: 0.01
```

---

## Scenario reference

Six ground-truth scenarios defined in `config.json`:

| Scenario | Dominant coefficient | Meaning |
|---|---|---|
| normal | a₄ = -2 | Nominal (mild bowl) |
| tilted | **a₂ = +8** | Chuck level error (Tilt X) |
| bowl | **a₄ = -12** | Center-to-edge imbalance (deep Defocus) |
| astigmatic | **a₆ = +6.5** | Showerhead 0/90 asymmetry |
| trefoil | **a₉ = +4.5** | 3-zone heater issue |
| comatic | **a₈ = +3.8** | X-direction asymmetric flow |

Total: 36 wafers = 6 scenarios + 30 drift samples (drift = a₁ decay + a₄ deepening + random walk on the others).

---

## Recipes

### Change noise strength

```bash
# Low noise (LSQ recovers near-perfectly)
python generate_samples.py --noise_sigma 0.4
python -m wlzpoly.decompose
python -m wlzpoly.verify

# High noise (LSQ wobbles, Ridge stabilizes)
python generate_samples.py --noise_sigma 10
python -m wlzpoly.decompose
python -m wlzpoly.verify
```

### Add a new scenario

Append to `config.json`:

```json
"scenarios": {
  ...
  "spherical_issue": {
    "1": 500.0, "2": 0.0, "3": 0.0, "4": -1.0,
    "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0
  }
}
```

No code change required.

### Extend the Zernike order

Edit `n_terms` in `config.json`:

```json
"decomposition": {
  "n_terms": 11,
  ...
}
```

(From 9 to 11 — note that 13 measurement points may not have enough degrees of freedom; aim for ≥ 4 residual DOF.)

### Change the measurement layout

Edit `points_13.json` directly (coordinates and zones are user-editable). No code change required.

### Write outputs to a different folder

```bash
python -m wlzpoly.verify --output_folder my_results
```

---

## Algorithm summary

### Zernike decomposition

Wafer thickness is modeled as a sum of Zernike polynomials on the unit disk:

```
T(ρ, θ) = Σ_{k=1..N} a_k · Z_k(ρ, θ) + ε
```

Sampled at the 13 measurement points, this becomes a linear system:

```
T = A · a + ε     (T: 13×1, A: 13×9, a: 9×1)
```

### Fitting

| Solver | Formula | Properties |
|---|---|---|
| **LSQ** | â = (AᵀA)⁻¹ Aᵀ T | Unbiased, higher variance |
| **Ridge** | â = (AᵀA + λI)⁻¹ Aᵀ T | Biased toward zero, lower variance |

λ is selected automatically by LOOCV (Leave-One-Out Cross-Validation).

### W2W / WiW separation

| Group | Coefficient(s) | Meaning |
|---|---|---|
| **W2W** (Wafer-to-Wafer) | a₁ (Piston) | Mean thickness |
| **WiW** (Within-Wafer) | a₂..a₉ | Spatial pattern (8 modes) |

Sanity check: 13 measurements → 9 coefficients → 4 residual DOF (used for residual monitoring).

---

## Dependencies

```
Python 3.9+
numpy >= 1.22
pandas >= 1.5
matplotlib >= 3.5
tqdm >= 4.60
```

Standard-library only beyond those: `argparse`, `csv`, `json`, `math`, `pathlib`, `typing`.

---

## Links

- **PyPI**: https://pypi.org/project/wlzpoly/
- **Source**: https://github.com/ykim2718/WaferLevelZernikePolynomials
- **Issues**: https://github.com/ykim2718/WaferLevelZernikePolynomials/issues
- **License**: MIT
