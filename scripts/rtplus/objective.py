"""Objective function for fitting RT+ to BF/DF loop-size data."""
from __future__ import annotations

from collections.abc import Mapping

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
    predicted_loop_log_intensity,
    predicted_loop_logpdf,
    predicted_observed_number_density,
    theta_for_image_visibility,
)
from .parameters import unpack_theta
from .simulation import simulate_all_temperatures
from .simulation import simulate_all_series


PENALTY = 1e100


def upper_trimmed_size_subset(values, retained_fraction=0.95):
    """Remove only the largest empirical tail from the robust size loss.

    This filtering applies only to the conditional diameter likelihood.  The
    smallest measured loops and the image count/volume term are all retained.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    retained_fraction = float(retained_fraction)
    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be in the interval (0, 1].")
    if values.size == 0 or retained_fraction == 1.0:
        return values, -np.inf, np.inf

    upper = float(np.quantile(values, retained_fraction))
    retained = values[values <= upper]
    if retained.size == 0:
        raise ValueError("Upper-tail size filtering removed every observation.")
    return retained, -np.inf, upper


def faulted_size_fit_fraction_for_prediction(prediction, fit_config):
    """Return the DF size fraction selected for one experimental event."""

    temperature_C = float(prediction["temperature_C"])
    full_temperatures = {
        float(value)
        for value in fit_config.faulted_full_distribution_temperatures
    }
    if any(
        np.isclose(temperature_C, selected, rtol=0.0, atol=1.0e-9)
        for selected in full_temperatures
    ):
        return 1.0
    return float(fit_config.faulted_size_fit_fraction)


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
    if counts.size == 0:
        raise ValueError("Image counts must be positive.")

    if isinstance(predicted_density, Mapping):
        predicted_densities = np.asarray(
            [
                float(predicted_density[str(image_id)])
                for image_id, _ in group.groupby("image", sort=True)
            ],
            dtype=float,
        )
    else:
        predicted_densities = np.full(
            counts.shape,
            float(predicted_density),
            dtype=float,
        )
    if (
        predicted_densities.shape != counts.shape
        or np.any(~np.isfinite(predicted_densities))
        or np.any(predicted_densities <= 0.0)
    ):
        raise ValueError("Predicted image densities must be positive and finite.")

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
    predicted_counts = predicted_densities * volumes
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
        event_loss_weight = (
            float(fit_config.room_temperature_loss_weight)
            if prediction.get("metadata", {}).get("simulated") is False
            else 1.0
        )
        faulted_size_fraction = faulted_size_fit_fraction_for_prediction(
            prediction,
            fit_config,
        )

        observation_config = ObservationConfig()
        if fit_config.objective_mode == "image_balanced_extended":
            image_losses = []
            predicted_densities = {}
            for image_id, image_data in group.groupby("image", sort=True):
                size_values_nm = image_data["size"].to_numpy(dtype=float)
                if str(mode).strip().upper() == "DF":
                    size_values_nm, _, _ = upper_trimmed_size_subset(
                        size_values_nm,
                        faulted_size_fraction,
                    )
                image_theta = theta_for_image_visibility(
                    theta,
                    series_id=series_id,
                    event_order=event_order,
                    mode=mode,
                    image_id=image_id,
                )
                predicted_density = predicted_observed_number_density(
                    mode=mode,
                    prediction=prediction,
                    theta=image_theta,
                    observation_config=observation_config,
                    radius_unit_to_nm=radius_unit_to_nm,
                )
                if predicted_density <= 0.0 or not np.isfinite(predicted_density):
                    return PENALTY
                predicted_densities[str(image_id)] = predicted_density
                log_intensity = predicted_loop_log_intensity(
                    values_nm=size_values_nm,
                    mode=mode,
                    prediction=prediction,
                    theta=image_theta,
                    radius_unit_to_nm=radius_unit_to_nm,
                    observation_config=observation_config,
                )
                logpdf = log_intensity - np.log(predicted_density)
                if len(logpdf) == 0 or not np.all(np.isfinite(logpdf)):
                    return PENALTY
                image_losses.append(-float(np.mean(logpdf)))

            absolute_count_modes = {
                str(value).strip().upper()
                for value in fit_config.absolute_count_modes
            }
            count_loss = 0.0
            if str(mode).strip().upper() in absolute_count_modes:
                try:
                    count_loss, _ = image_count_deviance(
                        group,
                        predicted_densities,
                        fit_config.count_overdispersion_floor,
                    )
                except ValueError:
                    return PENALTY

            dataset_loss = float(np.mean(image_losses)) + count_loss
            if not np.isfinite(dataset_loss):
                return PENALTY
            total_loss += event_loss_weight * dataset_loss
            number_of_contributions += 1

        elif fit_config.objective_mode == "legacy":
            values_nm = group["size"].to_numpy(dtype=float)
            if str(mode).strip().upper() == "DF":
                values_nm, _, _ = upper_trimmed_size_subset(
                    values_nm,
                    faulted_size_fraction,
                )
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
            dataset_loss += fit_config.density_loss_weight * density_loss
            total_loss += event_loss_weight * dataset_loss
            number_of_contributions += 1
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
