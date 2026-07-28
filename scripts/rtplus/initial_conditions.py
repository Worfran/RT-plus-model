"""Initial-condition strategies.

This file is intentionally isolated so you can change y0 without touching the
ODE, objective, optimization, or plotting code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import MaterialConstants
from .physics import (
    compute_radius,
    lognormal_rms_radius_from_mean,
    loop_content_from_radius,
)
from .parameters import faulted_width_at_temperature


def loguniform(low: float, high: float, rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


@dataclass(frozen=True)
class InitialConditionConfig:
    """y0 configuration.

    Supported strategies:
        random              : randomized physical y0.
        manual              : use manual_y0 exactly.
        from_nonirradiated  : use non-irradiated BF/DF mean diameters with placeholder densities.
    """

    strategy: str = "random"
    seed: Optional[int] = 10
    manual_y0: Optional[tuple[float, float, float, float, float, float]] = None

    Ci_range: tuple[float, float] = (1e-18, 1e-6)
    Cv0: float = 0.0
    Cf_range: tuple[float, float] = (1e14, 1e18)
    Cp_range: tuple[float, float] = (1e12, 1e18)
    Rf_nm_range: tuple[float, float] = (0.5, 10.0)
    Rp_nm_range: tuple[float, float] = (0.5, 20.0)

    # Used only by from_nonirradiated until real density data are added.
    data_based_Cf0: float = 1e16
    data_based_Cp0: float = 1e15
    data_based_Ci0: float = 1e-12


def make_initial_condition(
    ic_config: InitialConditionConfig,
    material: MaterialConstants,
    loop_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    strategy = ic_config.strategy.lower().strip()

    if strategy == "manual":
        if ic_config.manual_y0 is None:
            raise ValueError("manual_y0 must be provided for strategy='manual'.")
        y0 = np.asarray(ic_config.manual_y0, dtype=float)
        if y0.shape != (6,):
            raise ValueError("manual_y0 must have six entries: [Ci, Cv, Nf, Np, Cf, Cp].")
        return y0

    if strategy == "random":
        return _make_random_y0(ic_config, material)

    if strategy == "from_nonirradiated":
        if loop_data is None:
            raise ValueError("loop_data is required for strategy='from_nonirradiated'.")
        return _make_y0_from_nonirradiated(ic_config, material, loop_data)

    raise ValueError(f"Unknown initial-condition strategy: {ic_config.strategy}")


def _make_random_y0(ic_config: InitialConditionConfig, material: MaterialConstants) -> np.ndarray:
    rng = np.random.default_rng(ic_config.seed)

    Ci0 = loguniform(*ic_config.Ci_range, rng)
    Cv0 = ic_config.Cv0
    Cf0 = loguniform(*ic_config.Cf_range, rng)
    Cp0 = loguniform(*ic_config.Cp_range, rng)
    Rf0_nm = loguniform(*ic_config.Rf_nm_range, rng)
    Rp0_nm = loguniform(*ic_config.Rp_nm_range, rng)

    Rf0 = Rf0_nm * 1e-7
    Rp0 = Rp0_nm * 1e-7

    Nf0 = loop_content_from_radius(Rf0, Cf0, material.b, material.Omega0)
    Np0 = loop_content_from_radius(Rp0, Cp0, material.b, material.Omega0)

    return np.array([Ci0, Cv0, Nf0, Np0, Cf0, Cp0], dtype=float)


def _make_y0_from_nonirradiated(ic_config: InitialConditionConfig, material: MaterialConstants, loop_data: pd.DataFrame) -> np.ndarray:
    nonirr = loop_data[loop_data["irradiated"] == False].copy()
    if len(nonirr) == 0:
        raise ValueError("No non-irradiated data available for y0 estimation.")

    df_non = nonirr[nonirr["mode"] == "DF"]
    bf_non = nonirr[nonirr["mode"] == "BF"]

    Df_nm = float(df_non["size"].mean()) if len(df_non) else float(nonirr["size"].mean())
    Dp_nm = float(bf_non["size"].mean()) if len(bf_non) else Df_nm

    Rf0 = 0.5 * Df_nm * 1e-7
    Rp0 = 0.5 * Dp_nm * 1e-7
    Cf0 = ic_config.data_based_Cf0
    Cp0 = ic_config.data_based_Cp0
    Ci0 = ic_config.data_based_Ci0
    Cv0 = ic_config.Cv0

    Nf0 = loop_content_from_radius(Rf0, Cf0, material.b, material.Omega0)
    Np0 = loop_content_from_radius(Rp0, Cp0, material.b, material.Omega0)

    return np.array([Ci0, Cv0, Nf0, Np0, Cf0, Cp0], dtype=float)


def describe_y0(y0: np.ndarray, material: MaterialConstants) -> str:
    Ci, Cv, Nf, Np, Cf, Cp = y0
    Rf = compute_radius(Nf, Cf, material.b, material.Omega0)
    Rp = compute_radius(Np, Cp, material.b, material.Omega0)
    return (
        "Initial condition y0:\n"
        f"  Ci = {Ci:.3e}\n"
        f"  Cv = {Cv:.3e}\n"
        f"  Nf = {Nf:.3e}\n"
        f"  Np = {Np:.3e}\n"
        f"  Cf = {Cf:.3e} cm^-3\n"
        f"  Cp = {Cp:.3e} cm^-3\n"
        f"  Rf_rms = {Rf*1e7:.3f} nm\n"
        f"  Rp_rms = {Rp*1e7:.3f} nm"
    )

def make_series_initial_state(
    series_id,
    material,
    seed=None,
):
    """
    Build the initial state for one complete experimental series.

    This function is intentionally isolated so that the initial-condition
    method can later be replaced without changing the ODE or objective.
    """

    if series_id not in {"irradiated", "pristine"}:
        raise ValueError(f"Unknown series_id: {series_id}")

    series_seed = seed

    if seed is not None:
        if series_id == "irradiated":
            series_seed = seed
        elif series_id == "pristine":
            series_seed = seed + 1

    config = InitialConditionConfig(strategy="random", seed=series_seed)
    return _make_random_y0(config, material)


def fitted_initial_state(theta: dict, material: MaterialConstants) -> np.ndarray:
    """Construct a nonredundant fitted state in physical units.

    Ci0, Cv0, Cf0 and Cp0 are number concentrations in cm^-3.  The two
    stored-interstitial concentrations are derived from fitted arithmetic-mean
    radii, fitted distribution widths, and loop number densities.  This makes
    N proportional to the physical second moment C*<R^2>.
    """

    required = ("Ci0", "Cf0", "Cp0", "Rf0_nm", "Rp0_nm")
    missing = [name for name in required if name not in theta]
    if missing:
        raise KeyError(f"Missing fitted initial-condition parameters: {missing}")

    Ci0 = float(theta["Ci0"])
    Cv0 = float(theta.get("Cv0", 0.0))
    Cf0 = float(theta["Cf0"])
    Cp0 = float(theta["Cp0"])
    Rf0 = float(theta["Rf0_nm"]) * 1e-7
    Rp0 = float(theta["Rp0_nm"]) * 1e-7

    if min(Ci0, Cv0, Cf0, Cp0, Rf0, Rp0) < 0.0:
        raise ValueError("Fitted initial conditions must be nonnegative.")

    Nf0 = loop_content_from_radius(
        lognormal_rms_radius_from_mean(
            Rf0,
            faulted_width_at_temperature(theta),
        ),
        Cf0,
        material.b,
        material.Omega0,
    )
    Np0 = loop_content_from_radius(
        lognormal_rms_radius_from_mean(Rp0, theta["k_p"]),
        Cp0,
        material.b,
        material.Omega0,
    )
    return np.array([Ci0, Cv0, Nf0, Np0, Cf0, Cp0], dtype=float)


def fitted_initial_states(theta: dict, material: MaterialConstants, series_ids) -> dict:
    """Build fitted initial states for the currently fitted specimen series."""

    # The active workflow fits only the irradiated series.  Keeping the map
    # construction here makes the later addition of per-series IC parameters
    # explicit instead of silently sharing mutable state.
    state = fitted_initial_state(theta, material)
    return {series_id: state.copy() for series_id in series_ids}
