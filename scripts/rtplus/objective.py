"""Objective function for fitting RT+ to BF/DF loop-size data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FitConfig,
    MaterialConstants,
    ObservationConfig,
    SimulationConfig,
)
from .initial_conditions import fitted_initial_states
from .observables import (
    image_number_density_statistics,
    predicted_loop_logpdf,
    predicted_observed_number_density,
)
from .parameters import unpack_theta
from .simulation import simulate_all_temperatures
from .simulation import simulate_all_series


PENALTY = 1e100


def total_objective(
    theta_vector,
    loop_data,
    material,
    event_series,
    parameter_temperatures,
    fit_config,
    parameter_specs,
    radius_unit_to_nm=1e7,
):
    """
    Evaluate complete experimental histories.

    Events inside each series are sequential.
    BF and DF at one event use the same predicted material state.
    """

    theta = unpack_theta(
        theta_vector,
        parameter_temperatures,
        specs=parameter_specs,
    )

    initial_states = fitted_initial_states(theta, material, event_series)

    try:
        predictions = simulate_all_series(
            event_series=event_series,
            theta=theta,
            material=material,
            initial_states=initial_states,
        )
    except Exception:
        return PENALTY

    total_loss = 0.0
    number_of_contributions = 0

    fitted_series = set(event_series)
    data_to_fit = loop_data[loop_data["series_id"].isin(fitted_series)].copy()
    if fit_config.exclude_room_temperature:
        data_to_fit = data_to_fit[data_to_fit["temperature_C"] != 25.0]
    if fit_config.fit_temperatures is not None:
        selected = {float(T) for T in fit_config.fit_temperatures}
        data_to_fit = data_to_fit[data_to_fit["temperature_C"].isin(selected)]

    grouped_data = data_to_fit.groupby(
        ["series_id", "event_order", "mode"],
        sort=True,
    )

    for (
        series_id,
        event_order,
        mode,
    ), group in grouped_data:

        if series_id not in predictions:
            return PENALTY

        if event_order not in predictions[series_id]:
            return PENALTY

        prediction = predictions[series_id][event_order]

        values_nm = group["size"].to_numpy(dtype=float)

        logpdf = predicted_loop_logpdf(
            values_nm=values_nm,
            mode=mode,
            prediction=prediction,
            fit_theta=theta,
            radius_unit_to_nm=radius_unit_to_nm,
            observation_config=ObservationConfig(),
        )

        if len(logpdf) == 0:
            return PENALTY

        if not np.all(np.isfinite(logpdf)):
            return PENALTY

        dataset_loss = -np.mean(logpdf)

        if not np.isfinite(dataset_loss):
            return PENALTY

        total_loss += dataset_loss
        number_of_contributions += 1

        if series_id == "irradiated":
            observed_density, observed_std = image_number_density_statistics(
                group["image"].to_numpy(),
                group["volume_cm3"].to_numpy(dtype=float),
            )
            predicted_density = predicted_observed_number_density(
                mode=mode,
                prediction=prediction,
                theta=theta,
                observation_config=ObservationConfig(),
                radius_unit_to_nm=radius_unit_to_nm,
            )
            if predicted_density <= 0.0 or not np.isfinite(predicted_density):
                return PENALTY

            relative_std = observed_std / observed_density
            relative_std = max(
                relative_std,
                fit_config.density_relative_uncertainty_floor,
            )
            sigma_log = np.sqrt(np.log1p(relative_std**2))
            log_residual = np.log(predicted_density / observed_density)
            density_loss = 0.5 * (log_residual / sigma_log) ** 2
            total_loss += fit_config.density_loss_weight * density_loss

    if number_of_contributions == 0:
        return PENALTY

    if not np.isfinite(total_loss):
        return PENALTY

    return total_loss

def total_objective_v1(
    theta_vec,
    loop_data: pd.DataFrame,
    temperatures,
    material: MaterialConstants,
    sim_config: SimulationConfig,
    fit_config: FitConfig,
    y0: np.ndarray,
) -> float:
    try:
        theta = unpack_theta(theta_vec, temperatures)
        predictions = simulate_all_temperatures(temperatures, theta, material, sim_config, y0)

        data_to_fit = loop_data[
            (loop_data["irradiated"] == True) &
            (loop_data["temperature_C"].isin([float(T) for T in temperatures]))
        ].copy()

        if len(data_to_fit) == 0:
            return fit_config.objective_fail_value

        total_nll = 0.0
        n_contributions = 0

        for (T_C, mode), group in data_to_fit.groupby(["temperature_C", "mode"]):
            logpdf = predicted_loop_logpdf(
                values_nm=group["size"].to_numpy(dtype=float),
                mode=mode,
                prediction=predictions[float(T_C)],
                theta=theta,
            )
            if len(logpdf) == 0 or not np.all(np.isfinite(logpdf)):
                return fit_config.objective_fail_value

            # Mean contribution prevents datasets with more measured loops from dominating solely by count.
            total_nll += -float(np.mean(logpdf))
            n_contributions += 1

        if n_contributions == 0 or not np.isfinite(total_nll):
            return fit_config.objective_fail_value

        return float(total_nll)

    except Exception:
        return fit_config.objective_fail_value
