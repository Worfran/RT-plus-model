"""Objective function for fitting RT+ to BF/DF loop-size data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy

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


def _negative_binomial_logpmf(counts, means, overdispersion):
    """NB2 log-PMF with Var(N) = mean + overdispersion * mean**2."""

    counts = np.asarray(counts, dtype=float)
    means = np.asarray(means, dtype=float)
    alpha = float(overdispersion)
    if alpha <= 0.0 or np.any(counts < 0.0) or np.any(means < 0.0):
        raise ValueError("Counts, means, and overdispersion must be nonnegative.")

    shape = 1.0 / alpha
    denominator = shape + means
    return (
        gammaln(counts + shape)
        - gammaln(shape)
        - gammaln(counts + 1.0)
        + shape * (np.log(shape) - np.log(denominator))
        + xlogy(counts, means / denominator)
    )


def image_count_deviance(
    group,
    predicted_density,
    overdispersion_floor=0.04,
):
    """Return mean per-image negative-binomial deviance and dispersion.

    The empirical dispersion absorbs real image-to-image heterogeneity that is
    much larger than Poisson counting noise in several datasets.  The returned
    loss is measured relative to a saturated model, so zero is a perfect count
    fit and datasets with different loop totals remain comparable.
    """

    counts = []
    volumes = []
    for _, image_data in group.groupby("image", sort=True):
        image_volumes = image_data["volume_cm3"].drop_duplicates().to_numpy(float)
        if (
            image_volumes.size != 1
            or not np.isfinite(image_volumes[0])
            or image_volumes[0] <= 0.0
        ):
            raise ValueError("Each image must have one positive sampled volume.")
        counts.append(float(len(image_data)))
        volumes.append(float(image_volumes[0]))

    counts = np.asarray(counts, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if counts.size == 0 or predicted_density <= 0.0:
        raise ValueError("Image counts and predicted density must be positive.")

    image_densities = counts / volumes
    if counts.size > 1:
        relative_variance = (
            np.var(image_densities, ddof=1) / np.mean(image_densities) ** 2
        )
        poisson_relative_variance = np.mean(1.0 / np.maximum(counts, 1.0))
        empirical_overdispersion = max(
            0.0,
            relative_variance - poisson_relative_variance,
        )
    else:
        empirical_overdispersion = 0.0

    alpha = max(float(overdispersion_floor), empirical_overdispersion)
    predicted_counts = float(predicted_density) * volumes
    fitted_logpmf = _negative_binomial_logpmf(
        counts,
        predicted_counts,
        alpha,
    )
    saturated_logpmf = _negative_binomial_logpmf(counts, counts, alpha)
    loss = float(np.mean(saturated_logpmf - fitted_logpmf))
    return max(loss, 0.0), alpha


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

        observation_config = ObservationConfig()
        if fit_config.objective_mode == "image_balanced_extended":
            image_losses = []
            for _, image_data in group.groupby("image", sort=True):
                logpdf = predicted_loop_logpdf(
                    values_nm=image_data["size"].to_numpy(dtype=float),
                    mode=mode,
                    prediction=prediction,
                    fit_theta=theta,
                    radius_unit_to_nm=radius_unit_to_nm,
                    observation_config=observation_config,
                )
                if len(logpdf) == 0 or not np.all(np.isfinite(logpdf)):
                    return PENALTY
                image_losses.append(-float(np.mean(logpdf)))

            predicted_density = predicted_observed_number_density(
                mode=mode,
                prediction=prediction,
                theta=theta,
                observation_config=observation_config,
                radius_unit_to_nm=radius_unit_to_nm,
            )
            if predicted_density <= 0.0 or not np.isfinite(predicted_density):
                return PENALTY
            try:
                count_loss, _ = image_count_deviance(
                    group,
                    predicted_density,
                    fit_config.count_overdispersion_floor,
                )
            except ValueError:
                return PENALTY

            dataset_loss = float(np.mean(image_losses)) + count_loss
            if not np.isfinite(dataset_loss):
                return PENALTY
            total_loss += dataset_loss
            number_of_contributions += 1

        elif fit_config.objective_mode == "legacy":
            values_nm = group["size"].to_numpy(dtype=float)
            logpdf = predicted_loop_logpdf(
                values_nm=values_nm,
                mode=mode,
                prediction=prediction,
                fit_theta=theta,
                radius_unit_to_nm=radius_unit_to_nm,
                observation_config=observation_config,
            )
            if len(logpdf) == 0 or not np.all(np.isfinite(logpdf)):
                return PENALTY
            dataset_loss = -float(np.mean(logpdf))
            if not np.isfinite(dataset_loss):
                return PENALTY
            total_loss += dataset_loss
            number_of_contributions += 1

            observed_density, observed_std = image_number_density_statistics(
                group["image"].to_numpy(),
                group["volume_cm3"].to_numpy(dtype=float),
            )
            predicted_density = predicted_observed_number_density(
                mode=mode,
                prediction=prediction,
                theta=theta,
                observation_config=observation_config,
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
        else:
            raise ValueError(
                "fit_config.objective_mode must be "
                "'image_balanced_extended' or 'legacy'."
            )

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
