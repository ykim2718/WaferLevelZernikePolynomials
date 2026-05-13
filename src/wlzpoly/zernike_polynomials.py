"""
zernike_polynomials.py
============================================================
Zernike polynomial basis (Noll convention).

Used by generate_samples.py, decompose.py, and verify.py to keep the
basis definition consistent across data generation and fitting.

Indexing
--------
Each basis Z_j is identified by Noll index j = 1, 2, 3, ....
The (n, m) pair is derived programmatically from j, so the class supports
arbitrary order.
============================================================
"""

import io
import math
from pathlib import Path
from typing import (
    Any, Dict, List, Literal, Optional, Tuple, Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .regression import (
    SOLVER_CHOICES,
    SolverLiteral,
    fit_lsq,
    fit_ridge,
)


CoeffsLike = Union[
    Dict[int, float], List[float], Tuple[float, ...], np.ndarray,
]
COORDINATE_CHOICES = ["cartesian", "polar"]
CoordinateLiteral = Literal["cartesian", "polar"]
PYRAMID_RETURN_TYPE_CHOICES = ["png", "figure"]
PyramidReturnTypeLiteral = Literal["png", "figure"]


class ZernikePolynomials:
    """Zernike polynomial basis on the unit disk.

    Two usage modes:

    1. **Pure math (class-level)**: index conversion, basis function
       evaluation, design matrix construction, and the Zernike pyramid
       visualization. Use class/staticmethods directly:
           ZernikePolynomials.basis_matrix(rho=..., theta=..., n_terms=9)
           ZernikePolynomials.nm_from_noll(j=4)
           ZernikePolynomials.pyramid_image(n_max=4, return_type='png')

    2. **A specific Zernike expansion (instance)**: holds a coefficient
       set and lets you evaluate that specific wavefront.
           z = ZernikePolynomials(coeffs={1: 500.0, 4: -2.0}, n_terms=9)
           field = z.evaluate(rho=rho, theta=theta)
           z.rms()              # wavefront RMS (excluding piston)
           z[4]                  # -> -2.0 (a_4)
           len(z)                # -> 9   (n_terms)
       For wafer-level rendering of a fitted wavefront see
       WaferLevelZernikePolynomials.draw_field().

    Noll convention (Robert J. Noll, 1976) is used for indexing
    (j = 1, 2, ...) and for the orthonormality factors
    sqrt(n+1) / sqrt(2(n+1)).

    Sign convention for trigonometric part:
        m > 0 -> cos(|m| * theta)
        m < 0 -> sin(|m| * theta)
    """

    # ----------------------------------------------------------
    # Instance lifecycle (a specific Zernike expansion)
    # ----------------------------------------------------------
    def __init__(
        self,
        *,
        coeffs: CoeffsLike,
        n_terms: int = 9,
    ) -> None:
        assert isinstance(n_terms, int), (
            f"n_terms must be int, got {type(n_terms).__name__}"
        )
        assert n_terms >= 1, f"n_terms must be >= 1, got {n_terms}"
        assert isinstance(
            coeffs, (dict, list, tuple, np.ndarray)
        ), (
            f"coeffs must be dict/list/tuple/ndarray, got "
            f"{type(coeffs).__name__}"
        )

        # Normalize to dict {j: a_j}
        if isinstance(coeffs, dict):
            self.coeffs: Dict[int, float] = {
                int(k): float(v) for k, v in coeffs.items()
            }
        else:
            arr = np.asarray(coeffs).ravel()
            self.coeffs = {
                int(i + 1): float(v) for i, v in enumerate(arr)
            }
        self.n_terms = n_terms

    def __getitem__(self, j: int) -> float:
        """z[j] -> coefficient a_j (0.0 if not set)."""
        assert isinstance(j, (int, np.integer)), (
            f"j must be int, got {type(j).__name__}"
        )
        return self.coeffs.get(int(j), 0.0)

    def __len__(self) -> int:
        """len(z) -> n_terms (expansion capacity, not # nonzero)."""
        return self.n_terms

    def __repr__(self) -> str:
        if not self.coeffs:
            return f"ZernikePolynomials(empty, n_terms={self.n_terms})"
        a1 = self.coeffs.get(1, 0.0)
        non_piston = {
            j: a for j, a in self.coeffs.items() if j != 1
        }
        if non_piston:
            dom_j = max(non_piston, key=lambda j: abs(non_piston[j]))
            dom_str = f", dominant=a{dom_j}={non_piston[dom_j]:+.3f}"
        else:
            dom_str = ""
        return (
            f"ZernikePolynomials(n_terms={self.n_terms}, "
            f"a1={a1:.3f}{dom_str})"
        )

    # ----------------------------------------------------------
    # Instance methods (use self.coeffs)
    # ----------------------------------------------------------
    def evaluate(
        self,
        *,
        rho: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate this Zernike expansion: Sum_j self[j] * Z_j."""
        assert isinstance(rho, np.ndarray), (
            f"rho must be np.ndarray, got {type(rho).__name__}"
        )
        assert isinstance(theta, np.ndarray), (
            f"theta must be np.ndarray, got {type(theta).__name__}"
        )
        field = np.zeros_like(rho, dtype=float)
        for j, a in self.coeffs.items():
            if a == 0.0:
                continue
            field = field + a * type(self).basis(
                j=j, rho=rho, theta=theta,
            )
        return field

    def rms(self, *, exclude_piston: bool = True) -> float:
        """Wavefront RMS = sqrt(sum a_j^2) (Noll-orthonormal basis).

        Piston (a_1) is excluded by default since it represents mean
        offset, not wavefront error.
        """
        assert isinstance(exclude_piston, bool), (
            f"exclude_piston must be bool, got "
            f"{type(exclude_piston).__name__}"
        )
        total = 0.0
        for j, a in self.coeffs.items():
            if exclude_piston and j == 1:
                continue
            total += a * a
        return float(math.sqrt(total))

    # ----------------------------------------------------------
    # Noll index -> (n, m) conversion
    # ----------------------------------------------------------
    @staticmethod
    def nm_from_noll(j: int) -> Tuple[int, int]:
        """Return (n, m) for a 1-based Noll index j.

        Standard Noll 1976 algorithm:
        - radial order n is smallest with (n+1)(n+2)/2 >= j
        - p = j - n(n+1)/2 - 1  (position 0..n in block)
        - |m| candidates: parity of m matches parity of n
          (n even -> 0,2,4,...,n ; n odd -> 1,3,5,...,n)
        - sign rule: even j -> cos (m >= 0), odd j -> sin (m <= 0)
        """
        assert isinstance(j, (int, np.integer)), (
            f"j must be int, got {type(j).__name__}"
        )
        if j < 1:
            raise ValueError("Noll index must be >= 1")

        # 1. radial order n
        n = 0
        while (n + 1) * (n + 2) // 2 < j:
            n += 1

        # 2. position p inside the n-block (0..n)
        p = j - n * (n + 1) // 2 - 1

        # 3. |m| sequence in Noll order, low to high
        if n % 2 == 0:
            abs_ms = list(range(0, n + 1, 2))
        else:
            abs_ms = list(range(1, n + 1, 2))

        # 4. expand to signed m: each |m|>0 occupies two slots
        #    (sin then cos), m=0 occupies one slot
        signed_ms = []
        for am in abs_ms:
            if am == 0:
                signed_ms.append(0)
            else:
                signed_ms.append(-am)   # sin slot
                signed_ms.append(+am)   # cos slot
        m = signed_ms[p]

        # 5. enforce Noll sign rule via j parity
        if m != 0:
            wants_cos = (j % 2 == 0)
            if wants_cos and m < 0:
                m = -m
            elif (not wants_cos) and m > 0:
                m = -m

        return n, m

    # ----------------------------------------------------------
    # Polynomial evaluation
    # ----------------------------------------------------------
    @staticmethod
    def radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Radial polynomial R_n^|m|(rho)."""
        assert isinstance(n, (int, np.integer)), (
            f"n must be int, got {type(n).__name__}"
        )
        assert isinstance(m, (int, np.integer)), (
            f"m must be int, got {type(m).__name__}"
        )
        assert isinstance(rho, np.ndarray), (
            f"rho must be np.ndarray, got {type(rho).__name__}"
        )
        m = abs(m)
        out = np.zeros_like(rho, dtype=float)
        for k in range((n - m) // 2 + 1):
            num = (-1) ** k * math.factorial(n - k)
            den = (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
            out += (num / den) * rho ** (n - 2 * k)
        return out

    @classmethod
    def basis(
        cls, j: int, rho: np.ndarray, theta: np.ndarray,
    ) -> np.ndarray:
        """Noll-normalized Zernike basis Z_j(rho, theta)."""
        assert isinstance(j, (int, np.integer)), (
            f"j must be int, got {type(j).__name__}"
        )
        assert isinstance(rho, np.ndarray), (
            f"rho must be np.ndarray, got {type(rho).__name__}"
        )
        assert isinstance(theta, np.ndarray), (
            f"theta must be np.ndarray, got {type(theta).__name__}"
        )
        n, m = cls.nm_from_noll(j=j)
        R = cls.radial(n=n, m=m, rho=rho)
        if m == 0:
            return math.sqrt(n + 1) * R
        norm = math.sqrt(2 * (n + 1))
        if m > 0:
            return norm * R * np.cos(m * theta)
        return norm * R * np.sin(-m * theta)

    # ----------------------------------------------------------
    # Coordinate conversion (Cartesian <-> polar)
    # ----------------------------------------------------------
    @staticmethod
    def to_polar(
        *,
        x: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cartesian (x, y) -> polar (r, theta in radians).

        Accepts numpy arrays, pandas Series, or scalars. Output is in
        the unit system Zernike basis evaluation expects (theta in
        radians; r in same units as input - the consumer normalizes
        rho = r / wafer_radius_mm before calling basis_matrix).
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        assert x_arr.shape == y_arr.shape, (
            f"x and y shape mismatch: {x_arr.shape} vs {y_arr.shape}"
        )
        r = np.sqrt(x_arr ** 2 + y_arr ** 2)
        theta = np.arctan2(y_arr, x_arr)
        return r, theta

    @staticmethod
    def to_cartesian(
        *,
        r: Union[np.ndarray, Any],
        theta: Union[np.ndarray, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Polar (r, theta in radians) -> Cartesian (x, y).

        Accepts numpy arrays, pandas Series, or scalars.
        """
        r_arr = np.asarray(r, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        assert r_arr.shape == theta_arr.shape, (
            f"r and theta shape mismatch: "
            f"{r_arr.shape} vs {theta_arr.shape}"
        )
        x = r_arr * np.cos(theta_arr)
        y = r_arr * np.sin(theta_arr)
        return x, y

    @classmethod
    def basis_matrix(
        cls,
        rho: np.ndarray,
        theta: np.ndarray,
        *,
        n_terms: int = 9,
    ) -> np.ndarray:
        """Build the basis matrix A.

        A[i, k] = Z_{k+1}(rho_i, theta_i)
        Shape: (len(rho), n_terms).
        """
        assert isinstance(rho, np.ndarray), (
            f"rho must be np.ndarray, got {type(rho).__name__}"
        )
        assert isinstance(theta, np.ndarray), (
            f"theta must be np.ndarray, got {type(theta).__name__}"
        )
        assert isinstance(n_terms, int), (
            f"n_terms must be int, got {type(n_terms).__name__}"
        )
        A = np.zeros((len(rho), n_terms))
        for k in range(n_terms):
            A[:, k] = cls.basis(j=k + 1, rho=rho, theta=theta)
        return A

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------
    @classmethod
    def pyramid_image(
        cls,
        *,
        n_max: int = 4,
        grid_n: int = 200,
        names: Optional[Dict[Tuple[int, int], str]] = None,
        cell_size: float = 1.6,
        cmap: str = "RdBu_r",
        return_type: PyramidReturnTypeLiteral = "png",
    ) -> Union[bytes, Figure]:
        """Render the canonical Zernike pyramid up to radial order n_max.

        Cells in row n are placed at azimuthal frequency
        m in (-n, -n+2, ..., +n) and packed flush horizontally. Each
        successive row is offset by half a cell width so its cells
        straddle the cells of the row above.

        Parameters
        ----------
        n_max : int
            Highest radial order to draw (rows 0..n_max).
        grid_n : int
            Resolution of the unit-disk heatmap grid.
        names : dict {(n, m): str}, optional
            Optional human-readable labels per (n, m). Falls back to a
            "Z(n,m)" placeholder.
        cell_size : float
            Per-cell size in inches (cells are square).
        cmap : str
            Matplotlib colormap name.
        return_type : "png" or "figure"
            - "png"    : encode the figure to PNG bytes (default).
              Caller can write to file (`Path("p.png").write_bytes(b)`),
              embed in HTML, or display with IPython.display.Image.
            - "figure" : return the live matplotlib Figure for further
              customization. Caller must close / save themselves.

        Returns
        -------
        Union[bytes, matplotlib.figure.Figure]
            PNG bytes if return_type='png', Figure if 'figure'.
        """
        assert isinstance(return_type, str), (
            f"return_type must be str, got {type(return_type).__name__}"
        )
        assert return_type in PYRAMID_RETURN_TYPE_CHOICES, (
            f"return_type must be one of "
            f"{PYRAMID_RETURN_TYPE_CHOICES}, got {return_type!r}"
        )
        assert isinstance(n_max, int), (
            f"n_max must be int, got {type(n_max).__name__}"
        )
        assert n_max >= 0, f"n_max must be >= 0, got {n_max}"
        assert isinstance(grid_n, int), (
            f"grid_n must be int, got {type(grid_n).__name__}"
        )
        assert names is None or isinstance(names, dict), (
            f"names must be dict or None, got {type(names).__name__}"
        )
        assert isinstance(cell_size, (int, float)), (
            f"cell_size must be number, got {type(cell_size).__name__}"
        )
        assert isinstance(cmap, str), (
            f"cmap must be str, got {type(cmap).__name__}"
        )

        # ---- Build unit-disk evaluation grid ----
        ax_lin = np.linspace(-1, 1, grid_n)
        X, Y = np.meshgrid(ax_lin, ax_lin)
        RHO = np.sqrt(X ** 2 + Y ** 2)
        TH = np.arctan2(Y, X)
        mask = RHO <= 1.0

        # ---- Reverse lookup (n, m) -> Noll j for titles ----
        nm_to_j: Dict[Tuple[int, int], int] = {}
        j = 1
        max_terms = (n_max + 1) * (n_max + 2) // 2
        while j <= max_terms:
            n_j, m_j = cls.nm_from_noll(j=j)
            nm_to_j[(n_j, m_j)] = j
            j += 1

        # ---- Figure layout (manual add_axes for tight packing) ----
        # Each row: cells touch horizontally; (n_max+1) cells fit in
        # the bottom-most row, so figure width = (n_max+1)*cell_size.
        # Vertical: small gap above each row for title text.
        n_rows = n_max + 1
        title_gap_inches = cell_size * 0.09
        suptitle_inches = 0.71

        fig_w = cell_size * n_rows
        fig_h = (
            suptitle_inches
            + n_rows * (cell_size + title_gap_inches)
        )
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"Zernike pyramid (Noll convention, up to n={n_max})",
            fontsize=12,
            y=1.0 - 0.5 * suptitle_inches / fig_h,
        )

        cell_w_norm = cell_size / fig_w           # = 1 / n_rows
        cell_h_norm = cell_size / fig_h
        row_h_norm = (cell_size + title_gap_inches) / fig_h
        suptitle_norm = suptitle_inches / fig_h

        # ---- Fill pyramid cells ----
        for n in range(n_rows):
            cell_bottom = (
                1.0 - suptitle_norm - (n + 1) * row_h_norm
                + (row_h_norm - cell_h_norm)  # title gap goes above
            )
            for k, m in enumerate(range(-n, n + 1, 2)):
                cell_left = (k + (n_max - n) / 2.0) * cell_w_norm
                ax = fig.add_axes(
                    [cell_left, cell_bottom, cell_w_norm, cell_h_norm]
                )

                # Direct Noll-normalized evaluation on (n, m)
                R = cls.radial(n=n, m=m, rho=RHO)
                if m == 0:
                    Z = math.sqrt(n + 1) * R
                elif m > 0:
                    Z = math.sqrt(2 * (n + 1)) * R * np.cos(m * TH)
                else:
                    Z = math.sqrt(2 * (n + 1)) * R * np.sin(-m * TH)

                Z_masked = np.where(mask, Z, np.nan)
                vmax = float(np.nanmax(np.abs(Z_masked)))
                if vmax == 0.0:
                    vmax = 1.0
                ax.imshow(
                    Z_masked,
                    extent=[-1, 1, -1, 1],
                    origin="lower",
                    cmap=cmap,
                    vmin=-vmax, vmax=vmax,
                )
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])

                j_val = nm_to_j.get((n, m))
                title = f"j={j_val}  ({n},{m:+d})"
                if names is not None and (n, m) in names:
                    title += f"  {names[(n, m)]}"
                ax.set_title(title, fontsize=7, pad=1)

        if return_type == "figure":
            return fig

        # return_type == "png" - encode figure to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()


# =============================================================
# WaferLevelZernikePolynomials: ZP + measurement geometry
# =============================================================
class WaferLevelZernikePolynomials(ZernikePolynomials):
    """ZernikePolynomials bound to a specific wafer measurement layout.

    Extends `ZernikePolynomials` with knowledge of the measurement-point
    coordinates. Pre-computes the shared Zernike basis matrix A once at
    construction time so that downstream fitting calls don't have to
    rebuild it.

    Typical use
    -----------
    >>> coords_df = load_wafer_coordinates(
    ...     samples_folder=samples_folder, coordinate="cartesian",
    ... )
    >>> mesured_df = load_measured_data(samples_folder=samples_folder)
    >>> wlz = WaferLevelZernikePolynomials(
    ...     coords_df=coords_df,
    ...     coordinate="cartesian",
    ...     n_terms=9,
    ... )
    >>> fit_results = wlz.fit_coefficients(
    ...     mesured_df=mesured_df, solver="lsq",
    ... )
    """

    def __init__(
        self,
        *,
        coords_df: pd.DataFrame,
        coordinate: CoordinateLiteral = "cartesian",
        n_terms: int = 9,
    ) -> None:
        """Build the per-wafer basis matrix A from coords_df.

        Parameters
        ----------
        coords_df : pd.DataFrame
            Output of `decompose.load_wafer_coordinates(...)`.

            Index   : 'point_id' (str), e.g. 'P1'..'P13'
            Columns :
                - if coordinate='cartesian': ['x', 'y']  (mm)
                - if coordinate='polar':     ['r', 'theta']  (mm, deg)
            Required attrs :
                attrs['wafer_radius_mm'] : float
        coordinate : "cartesian" or "polar"
            Which fields of coords_df to read.
        n_terms : int
            Number of Zernike terms (Noll j = 1..n_terms).
        """
        assert isinstance(coords_df, pd.DataFrame), (
            f"coords_df must be pd.DataFrame, got "
            f"{type(coords_df).__name__}"
        )
        assert isinstance(coordinate, str), (
            f"coordinate must be str, got {type(coordinate).__name__}"
        )
        assert coordinate in COORDINATE_CHOICES, (
            f"coordinate must be one of {COORDINATE_CHOICES}, "
            f"got {coordinate!r}"
        )
        assert isinstance(n_terms, int), (
            f"n_terms must be int, got {type(n_terms).__name__}"
        )
        assert "wafer_radius_mm" in coords_df.attrs, (
            "coords_df.attrs must include 'wafer_radius_mm'"
        )

        # Initialize base instance state with empty coeffs (this
        # WaferLevel object is a fitting tool, not a wavefront).
        super().__init__(coeffs={}, n_terms=n_terms)

        self.coords_df = coords_df
        wafer_radius_mm = coords_df.attrs["wafer_radius_mm"]

        # Cartesian -> polar conversion if needed
        if coordinate == "cartesian":
            r, theta_rad = self.to_polar(
                x=coords_df["x"], y=coords_df["y"],
            )
        else:  # polar - stored theta is in degrees
            r = coords_df["r"].to_numpy(dtype=float)
            theta_rad = np.deg2rad(
                coords_df["theta"].to_numpy(dtype=float)
            )

        rho = r / wafer_radius_mm
        self.A: np.ndarray = self.basis_matrix(
            rho=rho, theta=theta_rad, n_terms=n_terms,
        )

    def fit_coefficients(
        self,
        *,
        mesured_df: pd.DataFrame,
        solver: SolverLiteral = "lsq",
        lam: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Fit Zernike coefficients for every wafer in mesured_df.

        Pure compute helper - no progress bar, no file I/O. Uses the
        pre-computed self.A built in __init__.

        Parameters
        ----------
        mesured_df : pd.DataFrame
            Long-format DataFrame as returned by
            decompose.load_measured_data().

            Index   : MultiIndex(['wafer_id', 'point_id'])
            Columns : ['T']  (measured thickness)

            Example
                                            T
                wafer_id point_id
                W_01     P1            500.50
                         P2            498.20
                         ...
                         P13           502.07
                W_02     P1            498.71
                         ...
                ...
                W_36     P13           486.40
        solver : "lsq" or "ridge"
            Linear regression solver to use per wafer.
        lam : float
            Ridge regularization strength
            (used only when solver='ridge').

        Returns
        -------
        list of dict
            One entry per wafer:
                {"id": <wafer_id>,
                 "coeffs": np.ndarray of length n_terms}
            Order matches mesured_df's wafer_id order.
        """
        assert isinstance(mesured_df, pd.DataFrame), (
            f"mesured_df must be pd.DataFrame, got "
            f"{type(mesured_df).__name__}"
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

        # Reindex T by coords_df's point order so A rows and T rows
        # correspond to the same physical points (mesured_df may be
        # sorted differently, e.g. alphabetically P1, P10, P11, ..., P9).
        point_order = list(self.coords_df.index)

        fit_results: List[Dict[str, Any]] = []
        wafer_ids = (
            mesured_df.index.get_level_values("wafer_id").unique()
        )
        for wafer_id in wafer_ids:
            T = (
                mesured_df.xs(wafer_id, level="wafer_id")["T"]
                .reindex(point_order)
                .to_numpy(dtype=float)
            )
            if solver == "lsq":
                coeffs = fit_lsq(A=self.A, T=T)
            else:
                coeffs = fit_ridge(A=self.A, T=T, lam=lam)
            fit_results.append({"id": wafer_id, "coeffs": coeffs})
        return fit_results

    def draw_field(
        self,
        *,
        coeffs: CoeffsLike,
        grid_n: int = 200,
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (5.0, 4.5),
        show_points: bool = True,
    ) -> Figure:
        """Render a single wavefront's unit-disk heatmap (one image).

        Evaluates the coefficient sum

            T(rho, theta) = Sum_j coeffs[j] * Z_j(rho, theta)

        on a Cartesian grid covering the unit disk, masks values
        outside rho <= 1, and shows it as a single heatmap. The
        measurement points stored in self.coords_df are scattered on
        top in normalized disk coordinates so the rendering is tied to
        this WaferLevel instance's geometry.

        About "wavefront"
        -----------------
        In Zernike usage the term "wavefront" means a 2D scalar field
        on the unit disk that can be expanded into the orthonormal
        Zernike basis. The class itself is just math - it does NOT
        know what physical quantity the values represent. Domain
        interpretations include:

        - Optics  : optical path difference / aberration
                    (waves, microns)
        - Semicon : wafer thickness profile (mm, nm) - this project
        - Adaptive optics : deformable-mirror surface deviation (nm)

        So `draw_field(coeffs=...)` produces a single heatmap of
        whichever scalar field the caller's coefficients happen to
        encode; only the colorbar units differ across domains.

        Parameters
        ----------
        coeffs : dict / list / tuple / ndarray
            Zernike coefficients defining the wavefront (same form
            accepted by ZernikePolynomials.__init__).
        grid_n : int
            Resolution of the unit-disk heatmap grid.
        cmap : str
            Matplotlib colormap name.
        title : str, optional
            Figure title (defaults to repr of the wavefront).
        figsize : (float, float)
            Figure size in inches.
        show_points : bool
            If True, scatter self.coords_df measurement points on top
            of the heatmap (in normalized disk coordinates).

        Returns
        -------
        matplotlib.figure.Figure
            Single-panel figure containing the wavefront heatmap. The
            caller is responsible for saving (e.g. ``fig.savefig(...)``)
            or displaying.
        """
        assert isinstance(coeffs, (dict, list, tuple, np.ndarray)), (
            f"coeffs must be dict/list/tuple/ndarray, got "
            f"{type(coeffs).__name__}"
        )
        assert isinstance(grid_n, int), (
            f"grid_n must be int, got {type(grid_n).__name__}"
        )
        assert isinstance(cmap, str), (
            f"cmap must be str, got {type(cmap).__name__}"
        )
        assert title is None or isinstance(title, str), (
            f"title must be str or None, got {type(title).__name__}"
        )
        assert isinstance(show_points, bool), (
            f"show_points must be bool, got "
            f"{type(show_points).__name__}"
        )

        # Use a base ZernikePolynomials instance to evaluate the field.
        z = ZernikePolynomials(coeffs=coeffs, n_terms=self.n_terms)

        ax_lin = np.linspace(-1, 1, grid_n)
        X, Y = np.meshgrid(ax_lin, ax_lin)
        RHO = np.sqrt(X ** 2 + Y ** 2)
        TH = np.arctan2(Y, X)
        mask = RHO <= 1.0

        field = z.evaluate(rho=RHO, theta=TH)
        field_masked = np.where(mask, field, np.nan)

        mean_val = z.coeffs.get(1, 0.0)
        max_dev = float(np.nanmax(np.abs(field_masked - mean_val)))
        if max_dev == 0.0:
            max_dev = 1.0
        vmin, vmax = mean_val - max_dev, mean_val + max_dev

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            field_masked,
            extent=[-1, 1, -1, 1],
            origin="lower",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
        )

        # Overlay measurement points from self.coords_df (normalized).
        if show_points:
            wafer_radius_mm = self.coords_df.attrs["wafer_radius_mm"]
            cols = set(self.coords_df.columns)
            if {"x", "y"}.issubset(cols):
                px = (
                    self.coords_df["x"].to_numpy(dtype=float)
                    / wafer_radius_mm
                )
                py = (
                    self.coords_df["y"].to_numpy(dtype=float)
                    / wafer_radius_mm
                )
            else:  # polar fields ('r', 'theta' in degrees)
                rr = (
                    self.coords_df["r"].to_numpy(dtype=float)
                    / wafer_radius_mm
                )
                th = np.deg2rad(
                    self.coords_df["theta"].to_numpy(dtype=float)
                )
                px = rr * np.cos(th)
                py = rr * np.sin(th)
            ax.scatter(px, py, c="k", s=14, zorder=3)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title if title is not None else repr(z), fontsize=10,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig
