"""
regression.py
============================================================
General-purpose linear regression solvers used by the Zernike fitting
workflow. Not Zernike-specific - operates on any (A, T) design matrix
and target vector.

Provided
--------
- fit_lsq(A, T)         : Plain least squares
- fit_ridge(A, T, lam)  : Ridge regression with L2 penalty lam
- loocv_lambda(A, T,
              lambdas)  : Leave-One-Out CV to pick best lam
============================================================
"""

from typing import Dict, List, Literal, Tuple

import numpy as np


SOLVER_CHOICES = ["lsq", "ridge"]
SolverLiteral = Literal["lsq", "ridge"]


# -------------------------------------------------------------
# Solvers
# -------------------------------------------------------------
def fit_lsq(A: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Plain Least Squares: a_hat = (A^T A)^-1 A^T T."""
    assert isinstance(A, np.ndarray), (
        f"A must be np.ndarray, got {type(A).__name__}"
    )
    assert isinstance(T, np.ndarray), (
        f"T must be np.ndarray, got {type(T).__name__}"
    )
    return np.linalg.solve(A.T @ A, A.T @ T)


def fit_ridge(
    A: np.ndarray, T: np.ndarray, *, lam: float,
) -> np.ndarray:
    """Ridge Regression: a_hat = (A^T A + lam*I)^-1 A^T T."""
    assert isinstance(A, np.ndarray), (
        f"A must be np.ndarray, got {type(A).__name__}"
    )
    assert isinstance(T, np.ndarray), (
        f"T must be np.ndarray, got {type(T).__name__}"
    )
    assert isinstance(lam, (int, float)), (
        f"lam must be number, got {type(lam).__name__}"
    )
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ T)


# -------------------------------------------------------------
# Hyperparameter tuning
# -------------------------------------------------------------
def loocv_lambda(
    A: np.ndarray,
    T: np.ndarray,
    *,
    lambdas: List[float],
) -> Tuple[float, Dict[float, float]]:
    """Leave-One-Out CV: pick lam minimizing held-out error squared.

    Returns
    -------
    best_lam : float
        Lambda value with the lowest mean held-out squared error.
    errors : dict {lam: mean_squared_error}
        Per-lambda LOOCV error for inspection / printing.
    """
    assert isinstance(A, np.ndarray), (
        f"A must be np.ndarray, got {type(A).__name__}"
    )
    assert isinstance(T, np.ndarray), (
        f"T must be np.ndarray, got {type(T).__name__}"
    )
    assert isinstance(lambdas, list), (
        f"lambdas must be list, got {type(lambdas).__name__}"
    )
    m = A.shape[0]
    errors: Dict[float, float] = {}
    for lam in lambdas:
        sse = 0.0
        for i in range(m):
            keep = np.ones(m, dtype=bool)
            keep[i] = False
            a_hat = fit_ridge(A=A[keep, :], T=T[keep], lam=lam)
            T_pred_i = A[i, :] @ a_hat
            sse += (T[i] - T_pred_i) ** 2
        errors[lam] = sse / m
    best = min(errors, key=errors.get)
    return best, errors
