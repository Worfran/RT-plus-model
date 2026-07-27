"""Observation model mapping RT+ predictions to BF/DF histogram data."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import lognorm

from .config import ObservationConfig
from .physics import lognormal_mean_radius_from_rms
from .simulation import Prediction


def effective_bf_faulted_visibility(
    theta: dict,
    observation_config: ObservationConfig | None = None,
) -> float:
    """Return crystallographic visibility times BF detection efficiency."""

    cfg = observation_config or ObservationConfig()
    eta_bf_f = float(theta.get("eta_bf_f", 1.0))
    if not 0.0 < eta_bf_f <= 1.0:
        raise ValueError("eta_bf_f must be in the interval (0, 1].")
    return float(cfg.bf_faulted_visibility * eta_bf_f)


def lognormal_shape_from_mean_std(mean: float, std: float) -> float:
    mean = max(float(mean), 1e-30)
    std = max(float(std), 1e-30)
    return float(np.sqrt(np.log(1.0 + (std / mean) ** 2)))


def lognormal_logpdf_from_mean_and_k(x, mean: float, k: float):
    x = np.asarray(x, dtype=float)
    mean = max(float(mean), 1e-30)
    k = max(float(k), 1e-8)
    std = k * mean
    sigma_logn = lognormal_shape_from_mean_std(mean, std)
    mu_logn = np.log(mean) - 0.5 * sigma_logn**2
    return lognorm.logpdf(x, s=sigma_logn, scale=np.exp(mu_logn))


def lognormal_survival_from_mean_and_k(cutoff, mean: float, k: float) -> float:
    """Fraction of a positive lognormal population above a detection limit."""

    if cutoff <= 0.0:
        return 1.0
    mean = max(float(mean), 1e-30)
    k = max(float(k), 1e-8)
    sigma_logn = lognormal_shape_from_mean_std(mean, k * mean)
    mu_logn = np.log(mean) - 0.5 * sigma_logn**2
    return float(lognorm.sf(float(cutoff), s=sigma_logn, scale=np.exp(mu_logn)))


def predicted_mean_radii_nm(
    prediction,
    theta: dict,
    radius_unit_to_nm: float = 1e7,
) -> tuple[float, float]:
    """Return arithmetic-mean radii consistent with stored second moments."""
    if isinstance(prediction, dict):
        Rf_rms, Rp_rms = prediction["Rf"], prediction["Rp"]
    else:
        Rf_rms, Rp_rms = prediction.Rf, prediction.Rp
    return (
        lognormal_mean_radius_from_rms(Rf_rms, theta["k_f"]) * radius_unit_to_nm,
        lognormal_mean_radius_from_rms(Rp_rms, theta["k_p"]) * radius_unit_to_nm,
    )


def predicted_mean_diameters_nm(
    prediction,
    theta: dict,
    radius_unit_to_nm: float = 1e7,
) -> tuple[float, float]:
    Rf_nm, Rp_nm = predicted_mean_radii_nm(prediction, theta, radius_unit_to_nm)
    return 2.0 * Rf_nm, 2.0 * Rp_nm


def predicted_observed_number_density(
    mode: str,
    prediction,
    theta: dict,
    observation_config: ObservationConfig | None = None,
    radius_unit_to_nm: float = 1e7,
) -> float:
    """Map physical loop densities to TEM-observable number density.

    This is the lognormal analogue of Bawane et al. Eqs. 2 and 5.  The paper
    used normal radius distributions; the present raw-diameter fit retains a
    positive lognormal family and evaluates the corresponding survival
    fraction above each radius-resolution limit.
    """

    cfg = observation_config or ObservationConfig()
    mode = str(mode).strip().upper()
    if isinstance(prediction, dict):
        Rf, Rp = prediction["Rf"], prediction["Rp"]
        Cf, Cp = prediction["Cf"], prediction["Cp"]
    else:
        Rf, Rp = prediction.Rf, prediction.Rp
        Cf, Cp = prediction.Cf, prediction.Cp

    Rf_nm, Rp_nm = predicted_mean_radii_nm(prediction, theta, radius_unit_to_nm)

    if mode == "DF":
        visible_f = 1.0
        if cfg.apply_resolution_cutoff:
            visible_f = lognormal_survival_from_mean_and_k(
                cfg.relrod_resolution_radius_nm, Rf_nm, theta["k_f"]
            )
        return float(cfg.relrod_faulted_visibility * Cf * visible_f)

    if mode == "BF":
        visible_bf_faulted = effective_bf_faulted_visibility(theta, cfg)
        visible_f = 1.0
        visible_p = 1.0
        if cfg.apply_resolution_cutoff:
            visible_f = lognormal_survival_from_mean_and_k(
                cfg.bf_resolution_radius_nm, Rf_nm, theta["k_f"]
            )
            visible_p = lognormal_survival_from_mean_and_k(
                cfg.bf_resolution_radius_nm, Rp_nm, theta["k_p"]
            )
        return float(
            visible_bf_faulted * Cf * visible_f
            + cfg.bf_perfect_visibility * Cp * visible_p
        )

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_log_intensity(
    values_nm,
    mode: str,
    prediction,
    theta: dict | None = None,
    radius_unit_to_nm: float = 1e7,
    fit_theta: dict | None = None,
    observation_config: ObservationConfig | None = None,
):
    """Return log observable intensity in cm^-3 nm^-1.

    Both the expected image counts and the conditional diameter distribution
    are derived from this intensity.  This prevents visibility or resolution
    corrections from being applied to only one part of the observation model.
    """

    theta = theta if theta is not None else fit_theta
    if theta is None:
        raise ValueError("theta or fit_theta must be provided")
    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm)]
    values_nm = values_nm[values_nm > 0]
    if len(values_nm) == 0:
        return np.array([])

    mode = str(mode).strip().upper()

    if isinstance(prediction, dict):
        Rf, Rp = prediction["Rf"], prediction["Rp"]
        Cf, Cp = prediction["Cf"], prediction["Cp"]
    else:
        Rf, Rp = prediction.Rf, prediction.Rp
        Cf, Cp = prediction.Cf, prediction.Cp

    k_f = theta["k_f"]
    k_p = theta["k_p"]

    Df_nm, Dp_nm = predicted_mean_diameters_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )

    logpdf_f = lognormal_logpdf_from_mean_and_k(values_nm, Df_nm, k_f)
    logpdf_p = lognormal_logpdf_from_mean_and_k(values_nm, Dp_nm, k_p)

    cfg = observation_config or ObservationConfig()
    Cf = max(float(Cf), 1e-300)
    Cp = max(float(Cp), 1e-300)

    if mode == "DF":
        log_intensity = (
            np.log(max(cfg.relrod_faulted_visibility, 1e-300))
            + np.log(Cf)
            + logpdf_f
        )
        if cfg.apply_resolution_cutoff:
            cutoff_diameter_nm = 2.0 * cfg.relrod_resolution_radius_nm
            log_intensity = np.where(
                values_nm >= cutoff_diameter_nm,
                log_intensity,
                -np.inf,
            )
        return log_intensity

    if mode == "BF":
        visible_bf_faulted = effective_bf_faulted_visibility(theta, cfg)
        log_intensity = logsumexp(
            np.vstack(
                [
                    np.log(max(visible_bf_faulted, 1e-300))
                    + np.log(Cf)
                    + logpdf_f,
                    np.log(max(cfg.bf_perfect_visibility, 1e-300))
                    + np.log(Cp)
                    + logpdf_p,
                ]
            ),
            axis=0,
        )
        if cfg.apply_resolution_cutoff:
            cutoff_diameter_nm = 2.0 * cfg.bf_resolution_radius_nm
            log_intensity = np.where(
                values_nm >= cutoff_diameter_nm,
                log_intensity,
                -np.inf,
            )
        return log_intensity

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_logpdf(
    values_nm,
    mode: str,
    prediction,
    theta: dict | None = None,
    radius_unit_to_nm: float = 1e7,
    fit_theta: dict | None = None,
    observation_config: ObservationConfig | None = None,
):
    """Return the conditional diameter log-PDF for an observed loop."""

    theta = theta if theta is not None else fit_theta
    if theta is None:
        raise ValueError("theta or fit_theta must be provided")
    log_intensity = predicted_loop_log_intensity(
        values_nm=values_nm,
        mode=mode,
        prediction=prediction,
        theta=theta,
        radius_unit_to_nm=radius_unit_to_nm,
        observation_config=observation_config,
    )
    total_density = predicted_observed_number_density(
        mode=mode,
        prediction=prediction,
        theta=theta,
        observation_config=observation_config,
        radius_unit_to_nm=radius_unit_to_nm,
    )
    if total_density <= 0.0 or not np.isfinite(total_density):
        return np.full_like(log_intensity, -np.inf, dtype=float)
    return log_intensity - np.log(total_density)


def predicted_loop_pdf(
    x_nm,
    mode: str,
    prediction: Prediction,
    theta: dict,
    radius_unit_to_nm: float = 1e7,
    observation_config: ObservationConfig | None = None,
):
    return np.exp(
        predicted_loop_logpdf(
            x_nm,
            mode,
            prediction,
            theta,
            radius_unit_to_nm,
            observation_config=observation_config,
        )
    )


def binned_loop_number_density(
    values_nm,
    total_observed_density_cm3: float,
    bin_edges_nm,
) -> np.ndarray:
    """Return observed loop density per diameter for each histogram bin.

    For bin ``j`` this evaluates

        (n_j / n_total) * C_observed / Delta_D_j

    so the bin area is the observed loop concentration in that interval and
    the sum of all bin areas is ``C_observed`` when the edges include all
    measured diameters.
    """

    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm) & (values_nm > 0.0)]
    edges_nm = np.asarray(bin_edges_nm, dtype=float)

    if values_nm.size == 0:
        raise ValueError("At least one positive finite loop diameter is required.")
    if edges_nm.ndim != 1 or edges_nm.size < 2:
        raise ValueError("bin_edges_nm must contain at least two edges.")
    widths_nm = np.diff(edges_nm)
    if np.any(~np.isfinite(edges_nm)) or np.any(widths_nm <= 0.0):
        raise ValueError("bin_edges_nm must be finite and strictly increasing.")
    if total_observed_density_cm3 <= 0.0 or not np.isfinite(total_observed_density_cm3):
        raise ValueError("total_observed_density_cm3 must be positive and finite.")

    counts, _ = np.histogram(values_nm, bins=edges_nm)
    return (
        counts.astype(float)
        / float(values_nm.size)
        * float(total_observed_density_cm3)
        / widths_nm
    )


def image_number_density_statistics(
    image_ids,
    volume_cm3,
) -> tuple[float, float]:
    """Return mean and population standard deviation across TEM images."""

    image_ids = np.asarray(image_ids, dtype=str)
    volume_cm3 = np.asarray(volume_cm3, dtype=float)
    if image_ids.size == 0 or image_ids.size != volume_cm3.size:
        raise ValueError("Image IDs and volumes must be nonempty and have equal length.")

    densities = []
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        image_volumes = np.unique(volume_cm3[selected])
        if (
            image_volumes.size != 1
            or not np.isfinite(image_volumes[0])
            or image_volumes[0] <= 0.0
        ):
            raise ValueError(f"Image {image_id!r} must have one positive volume.")
        densities.append(float(np.count_nonzero(selected)) / float(image_volumes[0]))

    densities = np.asarray(densities, dtype=float)
    return float(np.mean(densities)), float(np.std(densities, ddof=0))


def binned_loop_number_density_from_images(
    values_nm,
    image_ids,
    volume_cm3,
    bin_edges_nm,
) -> np.ndarray:
    """Average per-image spectra; units are inverse volume per nm."""

    values_nm = np.asarray(values_nm, dtype=float)
    image_ids = np.asarray(image_ids, dtype=str)
    volume_cm3 = np.asarray(volume_cm3, dtype=float)
    edges_nm = np.asarray(bin_edges_nm, dtype=float)

    if not (values_nm.size == image_ids.size == volume_cm3.size):
        raise ValueError("Diameters, image IDs, and volumes must have equal length.")
    widths_nm = np.diff(edges_nm)
    if edges_nm.ndim != 1 or edges_nm.size < 2 or np.any(widths_nm <= 0.0):
        raise ValueError("bin_edges_nm must be strictly increasing.")

    spectra = []
    for image_id in np.unique(image_ids):
        selected = image_ids == image_id
        image_volumes = np.unique(volume_cm3[selected])
        if (
            image_volumes.size != 1
            or not np.isfinite(image_volumes[0])
            or image_volumes[0] <= 0.0
        ):
            raise ValueError(f"Image {image_id!r} must have one positive volume.")
        counts, _ = np.histogram(values_nm[selected], bins=edges_nm)
        spectra.append(counts.astype(float) / float(image_volumes[0]) / widths_nm)

    if not spectra:
        raise ValueError("At least one image is required.")
    return np.mean(np.vstack(spectra), axis=0)


def predicted_loop_number_density_distribution(
    x_nm,
    mode: str,
    prediction: Prediction,
    theta: dict,
    observation_config: ObservationConfig | None = None,
    radius_unit_to_nm: float = 1e7,
) -> np.ndarray:
    """Return the observable loop density spectrum in cm^-3 nm^-1."""

    return np.exp(
        predicted_loop_log_intensity(
            values_nm=x_nm,
            mode=mode,
            prediction=prediction,
            theta=theta,
            observation_config=observation_config,
            radius_unit_to_nm=radius_unit_to_nm,
        )
    )
