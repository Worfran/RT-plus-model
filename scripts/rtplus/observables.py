"""Observation model mapping RT+ predictions to BF/DF histogram data."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp
from scipy.stats import lognorm, norm, truncnorm

from .config import ObservationConfig
from .parameters import faulted_width_at_temperature
from .simulation import Prediction
from .visibility_calibration import visibility_image_key


_VISIBILITY_QUADRATURE_NODES, _VISIBILITY_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(64)
)
_VISIBILITY_QUANTILES = 0.5 * (_VISIBILITY_QUADRATURE_NODES + 1.0)
_VISIBILITY_QUADRATURE_WEIGHTS = 0.5 * _VISIBILITY_QUADRATURE_WEIGHTS


def effective_bf_faulted_visibility(
    theta: dict,
    observation_config: ObservationConfig | None = None,
) -> float:
    """Return the fixed crystallographic BF visibility of faulted loops."""

    cfg = observation_config or ObservationConfig()
    return float(cfg.bf_faulted_visibility)


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


def faulted_distribution_family(theta: dict, mode: str | None = None) -> str:
    """Return the selected faulted-loop observation family for one mode."""

    family_by_mode = theta.get("faulted_distribution_by_mode", {})
    normalized_mode = None if mode is None else str(mode).strip().upper()
    if normalized_mode in family_by_mode:
        family = str(family_by_mode[normalized_mode]).strip().lower()
    else:
        family = str(theta.get("faulted_distribution", "lognormal")).strip().lower()
    valid = {
        "normal",
        "lognormal",
        "truncated_normal",
        "zero_truncated_normal",
    }
    if family not in valid:
        raise ValueError(
            "Faulted distribution must be normal, lognormal, "
            "truncated_normal, or zero_truncated_normal."
        )
    return family


def _standard_normal_mills_ratio(t: float) -> float:
    return float(
        np.exp(
            -0.5 * float(t) ** 2
            - 0.5 * np.log(2.0 * np.pi)
            - log_ndtr(float(t))
        )
    )


def _truncated_normal_cv_from_t(t: float) -> float:
    mills_ratio = _standard_normal_mills_ratio(t)
    variance_factor = max(
        1.0 - float(t) * mills_ratio - mills_ratio**2,
        1.0e-15,
    )
    return float(np.sqrt(variance_factor) / (float(t) + mills_ratio))


@lru_cache(maxsize=4096)
def _truncated_normal_t_from_cv(k: float) -> float:
    """Return base-normal mean/std giving the requested truncated CV."""

    k = float(k)
    if not 0.0 < k < 0.98:
        raise ValueError(
            "A zero-truncated normal requires 0 < std/mean < 0.98."
        )
    return float(
        brentq(
            lambda t: _truncated_normal_cv_from_t(t) - k,
            -8.0,
            100.0,
        )
    )


def truncated_normal_parameters_from_mean_and_k(
    mean: float,
    k: float,
) -> tuple[float, float, float]:
    """Return ``loc``, ``scale``, and lower bound for a positive normal.

    The resulting zero-truncated distribution has arithmetic mean ``mean`` and
    coefficient of variation ``k``. This preserves the same physical
    mean/second-moment interpretation used by the lognormal closure.
    """

    mean = max(float(mean), 1.0e-30)
    k = max(float(k), 1.0e-8)
    t = _truncated_normal_t_from_cv(k)
    mills_ratio = _standard_normal_mills_ratio(t)
    scale = mean / (t + mills_ratio)
    location = t * scale
    lower_standardized = -location / scale
    return float(location), float(scale), float(lower_standardized)


def positive_centered_normal_parameters(
    center: float,
    k: float,
) -> tuple[float, float, float]:
    """Return paper-style Gaussian parameters and its standardized zero point.

    ``k = omega / center`` is the width ratio used by Bawane et al.  The
    Gaussian itself is *not* renormalized at zero: the published visibility
    moments integrate the ordinary Gaussian only over detectable positive
    radii.  The third return value is retained for diagnostics and backwards
    compatibility with callers that need the standardized zero location.
    """

    center = max(float(center), 1.0e-30)
    k = float(k)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("A centered normal requires a positive width ratio.")
    scale = k * center
    lower_standardized = -center / scale
    return center, scale, lower_standardized


def loop_size_logpdf(
    values,
    mean: float,
    k: float,
    family: str,
) -> np.ndarray:
    """Evaluate a loop-size family using the selected observation convention."""

    family = str(family).strip().lower()
    if family == "lognormal":
        return lognormal_logpdf_from_mean_and_k(values, mean, k)
    if family == "normal":
        location, scale, _ = positive_centered_normal_parameters(mean, k)
        logpdf = norm.logpdf(values, loc=location, scale=scale)
        return np.where(np.asarray(values, dtype=float) > 0.0, logpdf, -np.inf)
    if family == "zero_truncated_normal":
        location, scale, lower = positive_centered_normal_parameters(mean, k)
        return truncnorm.logpdf(
            values,
            a=lower,
            b=np.inf,
            loc=location,
            scale=scale,
        )
    if family == "truncated_normal":
        location, scale, lower = truncated_normal_parameters_from_mean_and_k(
            mean,
            k,
        )
        return truncnorm.logpdf(
            values,
            a=lower,
            b=np.inf,
            loc=location,
            scale=scale,
        )
    raise ValueError(f"Unknown loop-size distribution family: {family!r}")


def _loop_size_quantiles(
    mean: float,
    k: float,
    family: str,
) -> np.ndarray:
    if family == "lognormal":
        sigma_logn = lognormal_shape_from_mean_std(mean, k * mean)
        mu_logn = np.log(max(mean, 1.0e-30)) - 0.5 * sigma_logn**2
        return lognorm.ppf(
            _VISIBILITY_QUANTILES,
            s=sigma_logn,
            scale=np.exp(mu_logn),
        )
    if family == "normal":
        location, scale, _ = positive_centered_normal_parameters(mean, k)
        return norm.ppf(
            _VISIBILITY_QUANTILES,
            loc=location,
            scale=scale,
        )
    if family == "zero_truncated_normal":
        location, scale, lower = positive_centered_normal_parameters(mean, k)
        return truncnorm.ppf(
            _VISIBILITY_QUANTILES,
            a=lower,
            b=np.inf,
            loc=location,
            scale=scale,
        )
    if family == "truncated_normal":
        location, scale, lower = truncated_normal_parameters_from_mean_and_k(
            mean,
            k,
        )
        return truncnorm.ppf(
            _VISIBILITY_QUANTILES,
            a=lower,
            b=np.inf,
            loc=location,
            scale=scale,
        )
    raise ValueError(f"Unknown loop-size distribution family: {family!r}")


def _positive_loop_size_quadrature(
    mean: float,
    k: float,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-size nodes and unconditional probability weights."""

    if family != "normal":
        return (
            _loop_size_quantiles(mean, k, family),
            _VISIBILITY_QUADRATURE_WEIGHTS,
        )

    location, scale, _ = positive_centered_normal_parameters(mean, k)
    positive_fraction = float(norm.sf(0.0, loc=location, scale=scale))
    lower_quantile = 1.0 - positive_fraction
    quantiles = lower_quantile + positive_fraction * _VISIBILITY_QUANTILES
    diameters_nm = norm.ppf(quantiles, loc=location, scale=scale)
    return (
        diameters_nm,
        positive_fraction * _VISIBILITY_QUADRATURE_WEIGHTS,
    )


def visibility_parameters_for_mode(
    mode: str,
    theta: dict,
) -> tuple[float, float] | None:
    """Return fitted visibility radius and transition width for one mode."""

    mode = str(mode).strip().upper()
    if mode not in {"DF", "BF"}:
        raise ValueError(f"Unknown mode: {mode}")
    radius_key = f"Rvis_{mode}_nm"
    transition_key = f"dRvis_{mode}_nm"
    if radius_key not in theta and transition_key not in theta:
        return None
    if radius_key not in theta or transition_key not in theta:
        raise KeyError(
            f"Both {radius_key} and {transition_key} are required."
        )
    radius = float(theta[radius_key])
    transition = float(theta[transition_key])
    if radius < 0.0 or transition <= 0.0:
        raise ValueError("Visibility radius must be nonnegative and width positive.")
    return radius, transition


def theta_for_image_visibility(
    theta: dict,
    *,
    series_id,
    event_order,
    mode,
    image_id,
) -> dict:
    """Return a theta copy with one image's threshold and width active."""

    image_thresholds = theta.get("image_visibility_rvis_nm", {})
    if not image_thresholds:
        return theta
    key = visibility_image_key(
        series_id,
        event_order,
        mode,
        image_id,
    )
    if key not in image_thresholds:
        return theta
    mode = str(mode).strip().upper()
    local_theta = dict(theta)
    local_theta[f"Rvis_{mode}_nm"] = float(image_thresholds[key])
    image_widths = theta.get("image_visibility_drvis_nm", {})
    if key in image_widths:
        local_theta[f"dRvis_{mode}_nm"] = float(image_widths[key])
    return local_theta


def visibility_log_weight(values_nm, mode: str, theta: dict) -> np.ndarray:
    """Return log TEM detectability for measured loop diameters."""

    values_nm = np.asarray(values_nm, dtype=float)
    parameters = visibility_parameters_for_mode(mode, theta)
    if parameters is None:
        return np.where(values_nm > 0.0, 0.0, -np.inf)
    visibility_radius_nm, transition_radius_nm = parameters
    radii_nm = 0.5 * values_nm
    z = (radii_nm - visibility_radius_nm) / transition_radius_nm
    log_weight = -np.logaddexp(0.0, -z)
    return np.where(values_nm > 0.0, log_weight, -np.inf)


def visible_fraction_of_distribution(
    mean_diameter_nm: float,
    k: float,
    family: str,
    mode: str,
    theta: dict,
    hard_cutoff_diameter_nm: float | None = None,
) -> float:
    """Return the observable fraction of one loop-size distribution."""

    parameters = visibility_parameters_for_mode(mode, theta)
    if (
        parameters is None
        and hard_cutoff_diameter_nm is None
    ):
        if family == "normal":
            location, scale, _ = positive_centered_normal_parameters(
                mean_diameter_nm,
                k,
            )
            return float(norm.sf(0.0, loc=location, scale=scale))
        return 1.0
    if parameters is None and hard_cutoff_diameter_nm is not None:
        cutoff = float(hard_cutoff_diameter_nm)
        if family == "lognormal":
            sigma_logn = lognormal_shape_from_mean_std(
                mean_diameter_nm,
                k * mean_diameter_nm,
            )
            mu_logn = (
                np.log(max(float(mean_diameter_nm), 1.0e-30))
                - 0.5 * sigma_logn**2
            )
            return float(
                lognorm.sf(
                    cutoff,
                    s=sigma_logn,
                    scale=np.exp(mu_logn),
                )
            )
        if family == "truncated_normal":
            location, scale, lower = (
                truncated_normal_parameters_from_mean_and_k(
                    mean_diameter_nm,
                    k,
                )
            )
            return float(
                truncnorm.sf(
                    cutoff,
                    a=lower,
                    b=np.inf,
                    loc=location,
                    scale=scale,
                )
            )
        if family == "normal":
            location, scale, _ = positive_centered_normal_parameters(
                mean_diameter_nm,
                k,
            )
            return float(
                norm.sf(
                    cutoff,
                    loc=location,
                    scale=scale,
                )
            )

    diameters_nm, quadrature_weights = _positive_loop_size_quadrature(
        max(float(mean_diameter_nm), 1.0e-30),
        max(float(k), 1.0e-8),
        family,
    )
    weights = np.exp(visibility_log_weight(diameters_nm, mode, theta))
    if hard_cutoff_diameter_nm is not None:
        weights = np.where(
            diameters_nm >= float(hard_cutoff_diameter_nm),
            weights,
            0.0,
        )
    fraction = float(
        np.sum(quadrature_weights * weights)
    )
    return float(np.clip(fraction, 0.0, 1.0))


def visible_mean_diameter_of_distribution(
    mean_diameter_nm: float,
    k: float,
    family: str,
    mode: str,
    theta: dict,
    hard_cutoff_diameter_nm: float | None = None,
) -> float:
    """Return ``<D w(D)>/<w(D)>`` over observable positive diameters."""

    parameters = visibility_parameters_for_mode(mode, theta)
    if (
        parameters is None
        and hard_cutoff_diameter_nm is None
    ):
        if family == "normal":
            location, scale, _ = positive_centered_normal_parameters(
                mean_diameter_nm,
                k,
            )
            alpha = -location / scale
            positive_fraction = norm.sf(alpha)
            return float(
                location + scale * norm.pdf(alpha) / positive_fraction
            )
        return float(mean_diameter_nm)

    diameters_nm, quadrature_weights = _positive_loop_size_quadrature(
        max(float(mean_diameter_nm), 1.0e-30),
        max(float(k), 1.0e-8),
        family,
    )
    weights = np.exp(visibility_log_weight(diameters_nm, mode, theta))
    if hard_cutoff_diameter_nm is not None:
        weights = np.where(
            diameters_nm >= float(hard_cutoff_diameter_nm),
            weights,
            0.0,
        )
    denominator = float(
        np.sum(quadrature_weights * weights)
    )
    if denominator <= 0.0:
        return np.nan
    numerator = float(
        np.sum(
            quadrature_weights
            * diameters_nm
            * weights
        )
    )
    return numerator / denominator


def faulted_width_for_prediction(prediction, theta: dict) -> float:
    """Return the faulted-loop width associated with a predicted event."""

    if isinstance(prediction, dict):
        metadata = prediction.get("metadata", {})
        temperature_C = (
            None
            if metadata.get("simulated") is False
            else prediction.get("temperature_C")
        )
    else:
        temperature_C = getattr(prediction, "temperature_C", None)
    return faulted_width_at_temperature(theta, temperature_C)


def predicted_mean_radii_nm(
    prediction,
    theta: dict,
    radius_unit_to_nm: float = 1e7,
) -> tuple[float, float]:
    """Return the source RT+ representative radii used as distribution means."""
    if isinstance(prediction, dict):
        Rf, Rp = prediction["Rf"], prediction["Rp"]
    else:
        Rf, Rp = prediction.Rf, prediction.Rp
    return (
        float(Rf) * radius_unit_to_nm,
        float(Rp) * radius_unit_to_nm,
    )


def predicted_mean_diameters_nm(
    prediction,
    theta: dict,
    radius_unit_to_nm: float = 1e7,
) -> tuple[float, float]:
    Rf_nm, Rp_nm = predicted_mean_radii_nm(prediction, theta, radius_unit_to_nm)
    return 2.0 * Rf_nm, 2.0 * Rp_nm


def predicted_visible_mean_diameters_nm(
    mode: str,
    prediction,
    theta: dict,
    observation_config: ObservationConfig | None = None,
    radius_unit_to_nm: float = 1e7,
) -> tuple[float, float]:
    """Return visible component means after smooth TEM detectability."""

    cfg = observation_config or ObservationConfig()
    mode = str(mode).strip().upper()
    if mode not in {"DF", "BF"}:
        raise ValueError(f"Unknown mode: {mode}")
    Df_nm, Dp_nm = predicted_mean_diameters_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )
    k_f = faulted_width_for_prediction(prediction, theta)
    family_f = faulted_distribution_family(theta, mode)
    hard_cutoff_diameter_nm = None
    if cfg.apply_resolution_cutoff:
        hard_cutoff_diameter_nm = 2.0 * (
            cfg.relrod_resolution_radius_nm
            if mode == "DF"
            else cfg.bf_resolution_radius_nm
        )
    visible_Df_nm = visible_mean_diameter_of_distribution(
        Df_nm,
        k_f,
        family_f,
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    visible_Dp_nm = visible_mean_diameter_of_distribution(
        Dp_nm,
        theta["k_p"],
        "lognormal",
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    return visible_Df_nm, visible_Dp_nm


def predicted_observed_mean_diameter_nm(
    mode: str,
    prediction,
    theta: dict,
    observation_config: ObservationConfig | None = None,
    radius_unit_to_nm: float = 1e7,
) -> float:
    """Return the mean diameter of the complete observable DF/BF population."""

    cfg = observation_config or ObservationConfig()
    mode = str(mode).strip().upper()
    Df_nm, Dp_nm = predicted_mean_diameters_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )
    k_f = faulted_width_for_prediction(prediction, theta)
    family_f = faulted_distribution_family(theta, mode)
    hard_cutoff_diameter_nm = None
    if cfg.apply_resolution_cutoff:
        hard_cutoff_diameter_nm = 2.0 * (
            cfg.relrod_resolution_radius_nm
            if mode == "DF"
            else cfg.bf_resolution_radius_nm
        )

    visible_Df_nm = visible_mean_diameter_of_distribution(
        Df_nm,
        k_f,
        family_f,
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    fraction_f = visible_fraction_of_distribution(
        Df_nm,
        k_f,
        family_f,
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    if mode == "DF":
        return float(visible_Df_nm)
    if mode != "BF":
        raise ValueError(f"Unknown mode: {mode}")

    visible_Dp_nm = visible_mean_diameter_of_distribution(
        Dp_nm,
        theta["k_p"],
        "lognormal",
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    fraction_p = visible_fraction_of_distribution(
        Dp_nm,
        theta["k_p"],
        "lognormal",
        mode,
        theta,
        hard_cutoff_diameter_nm,
    )
    if isinstance(prediction, dict):
        Cf, Cp = float(prediction["Cf"]), float(prediction["Cp"])
    else:
        Cf, Cp = float(prediction.Cf), float(prediction.Cp)
    weight_f = effective_bf_faulted_visibility(theta, cfg) * Cf * fraction_f
    weight_p = cfg.bf_perfect_visibility * Cp * fraction_p
    denominator = weight_f + weight_p
    if denominator <= 0.0:
        return np.nan
    return float(
        (weight_f * visible_Df_nm + weight_p * visible_Dp_nm)
        / denominator
    )


def predicted_observed_number_density(
    mode: str,
    prediction,
    theta: dict,
    observation_config: ObservationConfig | None = None,
    radius_unit_to_nm: float = 1e7,
) -> float:
    """Map physical loop densities to TEM-observable number density.

    Crystallographic visibility and smooth size-dependent TEM detectability are
    separate factors. The latter integrates ``w(R)`` over the selected loop
    distribution and therefore changes only the observation model.
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
    k_f = faulted_width_for_prediction(prediction, theta)
    family_f = faulted_distribution_family(theta, mode)
    hard_cutoff_diameter_nm = None

    if mode == "DF":
        if cfg.apply_resolution_cutoff:
            hard_cutoff_diameter_nm = 2.0 * cfg.relrod_resolution_radius_nm
        visible_f = visible_fraction_of_distribution(
            2.0 * Rf_nm,
            k_f,
            family_f,
            mode,
            theta,
            hard_cutoff_diameter_nm,
        )
        return float(cfg.relrod_faulted_visibility * Cf * visible_f)

    if mode == "BF":
        visible_bf_faulted = effective_bf_faulted_visibility(theta, cfg)
        if cfg.apply_resolution_cutoff:
            hard_cutoff_diameter_nm = 2.0 * cfg.bf_resolution_radius_nm
        visible_f = visible_fraction_of_distribution(
            2.0 * Rf_nm,
            k_f,
            family_f,
            mode,
            theta,
            hard_cutoff_diameter_nm,
        )
        visible_p = visible_fraction_of_distribution(
            2.0 * Rp_nm,
            theta["k_p"],
            "lognormal",
            mode,
            theta,
            hard_cutoff_diameter_nm,
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

    k_f = faulted_width_for_prediction(prediction, theta)
    k_p = theta["k_p"]
    family_f = faulted_distribution_family(theta, mode)

    Df_nm, Dp_nm = predicted_mean_diameters_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )

    logpdf_f = loop_size_logpdf(values_nm, Df_nm, k_f, family_f)
    logpdf_p = loop_size_logpdf(values_nm, Dp_nm, k_p, "lognormal")
    log_detectability = visibility_log_weight(values_nm, mode, theta)

    cfg = observation_config or ObservationConfig()
    Cf = max(float(Cf), 1e-300)
    Cp = max(float(Cp), 1e-300)

    if mode == "DF":
        log_intensity = (
            np.log(max(cfg.relrod_faulted_visibility, 1e-300))
            + np.log(Cf)
            + logpdf_f
            + log_detectability
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
                    + logpdf_f
                    + log_detectability,
                    np.log(max(cfg.bf_perfect_visibility, 1e-300))
                    + np.log(Cp)
                    + logpdf_p
                    + log_detectability,
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
