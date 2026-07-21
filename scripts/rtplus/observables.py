"""Observation model mapping RT+ predictions to BF/DF histogram data."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import lognorm

from .config import ObservationConfig
from .physics import lognormal_mean_radius_from_rms
from .simulation import Prediction


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

    Rf_nm, Rp_nm = predicted_mean_radii_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )

    if mode == "DF":
        visible_f = lognormal_survival_from_mean_and_k(
            cfg.relrod_resolution_radius_nm, Rf_nm, theta["k_f"]
        )
        return float(cfg.relrod_faulted_visibility * Cf * visible_f)

    if mode == "BF":
        visible_f = lognormal_survival_from_mean_and_k(
            cfg.bf_resolution_radius_nm, Rf_nm, theta["k_f"]
        )
        visible_p = lognormal_survival_from_mean_and_k(
            cfg.bf_resolution_radius_nm, Rp_nm, theta["k_p"]
        )
        return float(
            cfg.bf_faulted_visibility * Cf * visible_f
            + cfg.bf_perfect_visibility * Cp * visible_p
        )

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_logpdf(values_nm, mode: str, prediction, theta: dict | None = None, radius_unit_to_nm: float = 1e7, fit_theta: dict | None = None, observation_config: ObservationConfig | None = None):
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

    if mode == "DF":
        return logpdf_f

    if mode == "BF":
        cfg = observation_config or ObservationConfig()
        Cf = max(float(Cf), 1e-300)
        Cp = max(float(Cp), 1e-300)
        visible_Cf = cfg.bf_faulted_visibility * Cf
        visible_Cp = cfg.bf_perfect_visibility * Cp
        wF = visible_Cf / (visible_Cf + visible_Cp)
        wP = visible_Cp / (visible_Cf + visible_Cp)
        return logsumexp(np.vstack([np.log(wF) + logpdf_f, np.log(wP) + logpdf_p]), axis=0)

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_pdf(x_nm, mode: str, prediction: Prediction, theta: dict, radius_unit_to_nm: float = 1e7):
    return np.exp(predicted_loop_logpdf(x_nm, mode, prediction, theta, radius_unit_to_nm))
