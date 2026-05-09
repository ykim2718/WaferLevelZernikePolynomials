# wlzpoly — Wafer-Level Zernike Polynomials

Decompose 13-point wafer thickness measurements into 9 Zernike coefficients (LSQ / Ridge), with a reproducible demo workflow that generates synthetic data, fits it, and verifies the recovered coefficients against ground truth.

## Install

```bash
pip install wlzpoly
```

Requires Python 3.9+. Dependencies: `numpy >= 1.22`, `pandas >= 1.5`, `matplotlib >= 3.5`, `tqdm >= 4.60`.

## Quick start

```python
import numpy as np
from wlzpoly import ZernikePolynomials, WaferLevelZernikePolynomials

# 1) Build a wavefront from known coefficients (Noll j -> a_j)
z = ZernikePolynomials(coeffs={1: 500.0, 4: -12.0, 6: 0.5}, n_terms=9)
field = z.evaluate(rho=np.array([0.0, 0.5, 1.0]),
                   theta=np.array([0.0, 0.0, 0.0]))

# 2) Fit Zernike coefficients from measurements at known coordinates
#    coords_df   : DataFrame indexed by point_id, columns ['x','y'] (mm),
#                  attrs['wafer_radius_mm']
#    df_measured : DataFrame indexed by MultiIndex(wafer_id, point_id),
#                  column ['T']
wlz = WaferLevelZernikePolynomials(
    coords_df=coords_df, coordinate="cartesian", n_terms=9,
)
fit_results = wlz.fit_coefficients(mesured_df=df_measured, solver="lsq")
# fit_results : list of {"id": <wafer_id>, "coeffs": np.ndarray}

# 3) Render a fitted wafer field
fig = wlz.draw_field(coeffs=fit_results[0]["coeffs"])
fig.savefig("W_01_fit.png", dpi=130, bbox_inches="tight")
```

`ZernikePolynomials` follows the **Noll convention** and supports any radial order — the j -> (n, m) mapping is computed dynamically.

## Public API

```python
from wlzpoly import (
    ZernikePolynomials,           # pure-math + per-wavefront instance
    WaferLevelZernikePolynomials, # wafer-aware (coords + measurements -> fit)
    fit_lsq, fit_ridge,           # general-purpose linear solvers
)
```

| Function / class | Purpose |
|---|---|
| `ZernikePolynomials.basis(j, rho, theta)` | Single Zernike basis Z_j(rho, theta) |
| `ZernikePolynomials.basis_matrix(rho, theta, n_terms=...)` | Design matrix A for fitting |
| `ZernikePolynomials.pyramid_image(n_max=..., names=..., return_type=...)` | Zernike pyramid PNG / Figure |
| `ZernikePolynomials(coeffs=...).evaluate(rho, theta)` | Evaluate a specific wavefront |
| `WaferLevelZernikePolynomials(coords_df, coordinate, n_terms)` | Pre-compute A from measurement layout |
| `wlz.fit_coefficients(mesured_df, solver, lam)` | Per-wafer LSQ / Ridge fit |
| `wlz.draw_field(coeffs)` | Heatmap with measurement-point overlay |
| `fit_lsq(A, T)` | a_hat = (A^T A)^-1 A^T T |
| `fit_ridge(A, T, lam)` | a_hat = (A^T A + lam I)^-1 A^T T |
| `loocv_lambda(A, T, lambdas)` | LOOCV-driven lambda selection |

## Three-stage demo (after development install)

```bash
git clone https://github.com/ykim2718/WaferLevelZernikePolynomials.git
cd WaferLevelZernikePolynomials
pip install -e .

cd examples
python generate_samples.py                       # Stage 1: synthesize wafers
python -m wlzpoly.decompose --solver lsq         # Stage 2: fit coefficients
python -m wlzpoly.verify --solver lsq ridge      # Stage 3: compare vs truth
```

Demo outputs land in `examples/{samples,decomposition,verification}/`. Pre-generated copies are visible on the GitHub repo.

## Documentation

Full documentation — folder layout, CLI options, configuration files, output schemas, scenario reference, recipes, algorithm summary — lives on the GitHub README:

**https://github.com/ykim2718/WaferLevelZernikePolynomials**

## Links

- **Source / docs**: https://github.com/ykim2718/WaferLevelZernikePolynomials
- **Issues**: https://github.com/ykim2718/WaferLevelZernikePolynomials/issues
- **License**: MIT
