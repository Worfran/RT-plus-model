"""Observation model mapping RT+ predictions to BF/DF histogram data."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import lognorm

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


def predicted_loop_logpdf(values_nm, mode: str, prediction, theta: dict | None = None, radius_unit_to_nm: float = 1e7, fit_theta: dict | None = None):
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

    Df_nm = 2.0 * Rf * radius_unit_to_nm
    Dp_nm = 2.0 * Rp * radius_unit_to_nm

    k_f = theta["k_f"]
    k_p = theta["k_p"]

    logpdf_f = lognormal_logpdf_from_mean_and_k(values_nm, Df_nm, k_f)
    logpdf_p = lognormal_logpdf_from_mean_and_k(values_nm, Dp_nm, k_p)

    if mode == "DF":
        return logpdf_f

    if mode == "BF":
        Cf = max(float(Cf), 1e-300)
        Cp = max(float(Cp), 1e-300)
        wF = Cf / (Cf + Cp)
        wP = Cp / (Cf + Cp)
        return logsumexp(np.vstack([np.log(wF) + logpdf_f, np.log(wP) + logpdf_p]), axis=0)

    raise ValueError(f"Unknown mode: {mode}")


def predicted_loop_pdf(x_nm, mode: str, prediction: Prediction, theta: dict, radius_unit_to_nm: float = 1e7):
    return np.exp(predicted_loop_logpdf(x_nm, mode, prediction, theta, radius_unit_to_nm))
