"""Optimization and first-level parallel multi-start fitting."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import FitConfig, MaterialConstants
from .objective import total_objective
from .parameters import build_theta0_and_bounds, randomize_theta0, unpack_theta


@dataclass(frozen=True)
class FitResult:
    success: bool
    message: str
    objective: float
    theta_vec: np.ndarray
    theta: dict
    start_index: int


def run_single_start(
    start_index: int,
    theta0: np.ndarray,
    bounds,
    loop_data: pd.DataFrame,
    material: MaterialConstants,
    event_series,
    parameter_temperatures,
    fit_config: FitConfig,
    parameter_specs,
) -> FitResult:
    result = minimize(
        total_objective,
        theta0,
        args=(loop_data, material, event_series, parameter_temperatures, fit_config, parameter_specs),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": fit_config.maxiter, "ftol": fit_config.ftol, "gtol": fit_config.gtol},
    )

    theta = unpack_theta(result.x, parameter_temperatures, specs=parameter_specs)
    theta["faulted_distribution"] = fit_config.faulted_distribution
    if fit_config.apply_smooth_visibility:
        theta.update(
            {
                "Rvis_DF_nm": fit_config.Rvis_DF_nm,
                "dRvis_DF_nm": fit_config.dRvis_DF_nm,
                "Rvis_BF_nm": fit_config.Rvis_BF_nm,
                "dRvis_BF_nm": fit_config.dRvis_BF_nm,
                "image_visibility_rvis_nm": dict(
                    fit_config.image_visibility_rvis_nm
                ),
                "image_visibility_offsets_nm": dict(
                    fit_config.image_visibility_offsets_nm
                ),
                "image_visibility_efficiency": dict(
                    fit_config.image_visibility_efficiency
                ),
            }
        )
    return FitResult(
        success=bool(result.success),
        message=str(result.message),
        objective=float(result.fun),
        theta_vec=np.asarray(result.x, dtype=float),
        theta=theta,
        start_index=start_index,
    )


def make_start_vectors(temperatures, fit_config: FitConfig, parameter_specs):
    theta0, bounds = build_theta0_and_bounds(temperatures, specs=parameter_specs)
    rng = np.random.default_rng(fit_config.random_seed)
    starts = []
    for i in range(fit_config.n_starts):
        if i == 0:
            starts.append(theta0.copy())
        else:
            starts.append(randomize_theta0(theta0, bounds, rng))
    return starts, bounds


def run_multistart(
    loop_data: pd.DataFrame,
    material: MaterialConstants,
    event_series,
    parameter_temperatures,
    fit_config: FitConfig,
    parameter_specs,
):
    starts, bounds = make_start_vectors(parameter_temperatures, fit_config, parameter_specs)

    if fit_config.parallel_starts and len(starts) > 1:
        results = []
        with ProcessPoolExecutor(max_workers=fit_config.max_workers) as ex:
            futures = [
                ex.submit(
                    run_single_start,
                    i,
                    start,
                    bounds,
                    loop_data,
                    material,
                    event_series,
                    parameter_temperatures,
                    fit_config,
                    parameter_specs,
                )
                for i, start in enumerate(starts)
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda r: r.start_index)
    else:
        results = [
            run_single_start(i, start, bounds, loop_data, material, event_series, parameter_temperatures, fit_config, parameter_specs)
            for i, start in enumerate(starts)
        ]

    successful = [result for result in results if result.success]
    best = min(successful or results, key=lambda r: r.objective)
    return best, results
