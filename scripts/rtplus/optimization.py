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


INITIAL_POPULATION_PARAMETER_NAMES = ("Ci0", "Cv0", "Cf0", "Cp0")


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
    theta["faulted_distribution_by_mode"] = {
        "DF": fit_config.faulted_distribution_df,
        "BF": fit_config.faulted_distribution_bf,
    }
    theta["coalescence_model"] = fit_config.coalescence_model
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
                "image_visibility_drvis_nm": dict(
                    fit_config.image_visibility_drvis_nm
                ),
                "image_visibility_width_log_offsets": dict(
                    fit_config.image_visibility_width_log_offsets
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


def _initial_population_parameter_indices(temperatures, parameter_specs):
    """Map fitted initial concentration names to vector positions."""

    n_temperatures = len([float(value) for value in temperatures])
    indices = {}
    vector_index = 0
    for spec in parameter_specs:
        if spec.scope == "global":
            if spec.name in INITIAL_POPULATION_PARAMETER_NAMES:
                if spec.transform != "log":
                    raise ValueError(
                        f"Initial population {spec.name} must use a log transform."
                    )
                indices[spec.name] = vector_index
            vector_index += 1
        elif spec.scope == "per_temperature":
            vector_index += n_temperatures
        else:
            raise ValueError(f"Unknown parameter scope: {spec.scope}")
    return indices


def overshoot_initial_population_start(
    theta0,
    bounds,
    temperatures,
    parameter_specs,
    multiplier,
):
    """Scale existing fitted initial concentrations without adding parameters."""

    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("Initial-population start multipliers must be at least one.")

    start = np.asarray(theta0, dtype=float).copy()
    for vector_index in _initial_population_parameter_indices(
        temperatures,
        parameter_specs,
    ).values():
        low, high = bounds[vector_index]
        start[vector_index] = np.clip(
            start[vector_index] + np.log(multiplier),
            low,
            high,
        )
    return start


def make_start_vectors(temperatures, fit_config: FitConfig, parameter_specs):
    theta0, bounds = build_theta0_and_bounds(temperatures, specs=parameter_specs)
    population_indices = tuple(
        _initial_population_parameter_indices(
            temperatures,
            parameter_specs,
        ).values()
    )
    minimum_multiplier = float(fit_config.initial_population_min_multiplier)
    if not np.isfinite(minimum_multiplier) or minimum_multiplier < 1.0:
        raise ValueError(
            "initial_population_min_multiplier must be at least one."
        )
    constrained_bounds = list(bounds)
    for vector_index in population_indices:
        low, high = constrained_bounds[vector_index]
        minimum_value = theta0[vector_index] + np.log(minimum_multiplier)
        if minimum_value > high:
            raise ValueError(
                "Initial-population minimum multiplier exceeds the fitted "
                "parameter bound."
            )
        constrained_bounds[vector_index] = (max(low, minimum_value), high)
    bounds = constrained_bounds

    rng = np.random.default_rng(fit_config.random_seed)
    multipliers = tuple(
        float(value)
        for value in fit_config.initial_population_start_multipliers
    )
    if not multipliers or any(
        not np.isfinite(value) or value < 1.0
        for value in multipliers
    ):
        raise ValueError(
            "initial_population_start_multipliers must contain values >= 1."
        )

    anchors = [
        overshoot_initial_population_start(
            theta0,
            bounds,
            temperatures,
            parameter_specs,
            multiplier,
        )
        for multiplier in multipliers
    ]
    starts = []
    for i in range(fit_config.n_starts):
        anchor = anchors[i % len(anchors)]
        if i < len(anchors):
            # Guarantee that baseline, moderate, and strong population seeds
            # are evaluated before adding broader randomized starts.
            starts.append(anchor.copy())
        else:
            randomized = randomize_theta0(theta0, bounds, rng)
            # Preserve the selected population scale while continuing to
            # randomize every kinetic and distribution parameter broadly.
            for vector_index in population_indices:
                low, high = bounds[vector_index]
                randomized[vector_index] = np.clip(
                    anchor[vector_index] + rng.normal(0.0, np.log(2.0)),
                    low,
                    high,
                )
            starts.append(randomized)
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
