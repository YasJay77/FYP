"""Pore knockdown correction for the mechanical surrogate.

The surrogate is fit on pore-free homogenisation so has no d/L dependence; this
multiplies its output by a fitted knockdown k_ij(d/L; type), pinned to k(0)=1.
SC, FCC and FCC_face each get their own knockdown family (different pore topology),
which is what lets us distinguish FCC from FCC_face mechanically.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOCKDOWN_FITS_PATH = PROJECT_ROOT / "results" / "pore_investigation" / "knockdown_fits.json"


# knockdown forms, all give k(0)=1

def _k_power(x: np.ndarray, a: float, p: float) -> np.ndarray:
    """power-law knockdown 1 - a x^p, k(0)=1"""
    x = np.asarray(x, dtype=float)
    return 1.0 - a * np.power(np.maximum(x, 0.0), p)


def _k_quadratic(x: np.ndarray, b: float, c: float) -> np.ndarray:
    """quadratic knockdown 1 - b x - c x^2, k(0)=1"""
    x = np.asarray(x, dtype=float)
    return 1.0 - b * x - c * x * x


def _k_exp_power(x: np.ndarray, a: float, p: float) -> np.ndarray:
    """exponential knockdown exp(-a x^p), k(0)=1"""
    x = np.asarray(x, dtype=float)
    return np.exp(-a * np.power(np.maximum(x, 0.0), p))


_FORM_TO_FN = {
    "A_power_minus": _k_power,
    "B_quadratic": _k_quadratic,
    "C_exp_power": _k_exp_power,
}


# map FCC labels to canonical knockdown keys
_LATTICE_ALIASES = {
    "SC": "SC",
    "FCC": "FCC",
    "FCC_NODAL": "FCC",
    "FCC_NODAL_PORE": "FCC",
    "FCC_FACE": "FCC_face",
    "FCC_FACE_PORE": "FCC_face",
}


def _canonical_lattice(lattice_type: str) -> str:
    """map a lattice label to its canonical knockdown key"""
    key = lattice_type.upper().replace("-", "_").replace(" ", "_")
    if key not in _LATTICE_ALIASES:
        raise KeyError(
            f"Unknown lattice type {lattice_type!r}; "
            f"expected one of SC / FCC / FCC_face"
        )
    return _LATTICE_ALIASES[key]


class PoreKnockdown:
    """lazy-loaded pore knockdown evaluator"""

    def __init__(self, fits_path: Path | None = None):
        """load the fitted knockdowns and sanity-check the functional forms"""
        self.fits_path = Path(fits_path) if fits_path else KNOCKDOWN_FITS_PATH
        with open(self.fits_path) as f:
            self._fits = json.load(f)
        # bail early if any fit references an unknown form
        for lat, comps in self._fits.items():
            for comp, info in comps.items():
                if info["best"] not in _FORM_TO_FN:
                    raise ValueError(
                        f"{lat}/{comp}: unknown functional form {info['best']!r}"
                    )

    def factor(self, d_over_L, lattice_type: str, component: str) -> np.ndarray:
        """k_ij(d/L) for one component; vectorised, returns 1 at d/L=0"""
        lat = _canonical_lattice(lattice_type)
        key = component if component.startswith("k_") else f"k_{component}"
        info = self._fits[lat][key]
        fn = _FORM_TO_FN[info["best"]]
        return fn(np.asarray(d_over_L, dtype=float), *info["params"])

    def factors_all(self, d_over_L, lattice_type: str) -> dict:
        """{C11, C12, C44} knockdowns for one lattice family"""
        return {
            comp: self.factor(d_over_L, lattice_type, comp)
            for comp in ("C11", "C12", "C44")
        }

    def apply(self, C11, C12, C44, d_over_L, lattice_type: str):
        """knock down a pore-free stiffness triplet"""
        k = self.factors_all(d_over_L, lattice_type)
        return C11 * k["C11"], C12 * k["C12"], C44 * k["C44"]


def load_knockdown(fits_path: Path | None = None) -> PoreKnockdown:
    """convenience constructor for a PoreKnockdown"""
    return PoreKnockdown(fits_path=fits_path)


def predict_mechanical_corrected(
    t_over_L,
    d_over_L,
    lattice_type: str,
    ga_params: dict,
    knockdown: PoreKnockdown | None = None,
    E_material: float = 1.0,
):
    """surrogate query with the pore knockdown applied"""
    if knockdown is None:
        knockdown = load_knockdown()

    lat_kd = _canonical_lattice(lattice_type)
    # GA fit only knows 'SC'/'FCC', face-pore reuses the FCC fit
    lat_ga = "FCC" if lat_kd == "FCC_face" else lat_kd

    t = np.asarray(t_over_L, dtype=float)
    slope, intercept = ga_params[lat_ga]["rho_fit"]
    rho = np.clip(slope * t + intercept, 0.01, 1.0)

    out = {}
    for comp in ("C11", "C12", "C44"):
        C1, n = ga_params[lat_ga][comp]
        out[comp] = C1 * np.power(rho, n)

    C11_c, C12_c, C44_c = knockdown.apply(
        out["C11"], out["C12"], out["C44"], d_over_L, lattice_type
    )
    return C11_c * E_material, C12_c * E_material, C44_c * E_material


__all__ = [
    "PoreKnockdown",
    "load_knockdown",
    "predict_mechanical_corrected",
]
