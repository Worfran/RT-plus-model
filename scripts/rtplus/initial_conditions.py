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
from .physics import compute_radius, loop_content_from_radius


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
        f"  Rf = {Rf*1e7:.3f} nm\n"
        f"  Rp = {Rp*1e7:.3f} nm"
    )

import numpy as np


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
