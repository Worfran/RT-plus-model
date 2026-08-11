"""ODE system for the RT+ model."""
from __future__ import annotations

import numpy as np

from .physics import coalescence_number_loss, compute_radius, loop_flux


def rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """Right-hand side for y = [Ci, Cv, Nf, Np, Cf, Cp].

    The Frank-loop factor 1/3 is included in b, so the content relation is:
        N = (pi*b/Omega0)*R^2*C
    """

    if not np.all(np.isfinite(y)):
        raise FloatingPointError("Non-finite ODE state.")

    Ci, Cv, Nf, Np, Cf, Cp = y
    a = params["a"]
    Dv = params.get("Dv", 0.0)
    Ziv_iK = params.get("Ziv_iK", 0.0)
    Ziv_vK = params.get("Ziv_vK", 0.0)
    Di = params["Di"]
    Rii = params["Rii"]
    Puf = params["Puf"]
    Pcs = params["Pcs"]
    Pfcs = params["Pfcs"]
    b = params["b"]
    Omega0 = params["Omega0"]
    r0 = params["r0"]

    G0i = params.get("G0i", 0.0)
    G0v = params.get("G0v", 0.0)

    Ci_eff = max(float(Ci), 0.0)
    Cv_eff = max(float(Cv), 0.0)
    Nf_eff = max(float(Nf), 1e-300)
    Np_eff = max(float(Np), 1e-300)
    Cf_eff = max(float(Cf), 1e-300)
    Cp_eff = max(float(Cp), 1e-300)

    Rf = compute_radius(Nf_eff, Cf_eff, b, Omega0)
    Rp = compute_radius(Np_eff, Cp_eff, b, Omega0)

    jf = loop_flux(Rf, Di, Ci_eff, r0)
    jp = loop_flux(Rp, Di, Ci_eff, r0)

    geometry_factor = np.pi * b / Omega0
    enable_vacancy_extension = bool(params.get("enable_vacancy_extension", False))
    enable_surface_sink = bool(params.get("enable_surface_sink", False))
    kiv = 0.0
    if enable_vacancy_extension:
        kiv = (Omega0 / a**2) * (Ziv_iK * Di + Ziv_vK * Dv)

    surface_sink_strength = 0.0
    if enable_surface_sink:
        # For two absorbing planar surfaces, sink strength is 12/L^2.
        L = float(params["lamella_thickness_cm"])
        surface_sink_strength = 12.0 / L**2

    df = np.zeros(6)

    # Preserve the published S3-S6 convention: j_i^L contains R, and the
    # population balance uses R*C*j_i^L.  Applying the identical term with
    # opposite signs conserves interstitial content during loop absorption.
    faulted_absorption = Rf * Cf_eff * jf
    perfect_absorption = Rp * Cp_eff * jp

    # One di-interstitial nucleation event creates one faulted loop and moves
    # two interstitials from Ci into the stored faulted-loop content Nf.
    diinterstitial_nucleation = Rii * Di * Ci_eff**2

    # Interaction-driven coalescence from the loop-lifetime publication:
    #   1/tau = lambda(T)*R*C^(5/3)
    #   C/tau = lambda(T)*R*C^(8/3).
    # Coalescence removes loop objects without removing their stored inventory.
    faulted_coalescence_loss = coalescence_number_loss(
        Pfcs,
        Rf,
        Cf_eff,
        model=params.get("coalescence_model", "interaction_driven"),
    )
    perfect_coalescence_loss = coalescence_number_loss(
        Pcs,
        Rp,
        Cp_eff,
        model=params.get("coalescence_model", "interaction_driven"),
    )

    df[0] = G0i - faulted_absorption - perfect_absorption - 2.0 * diinterstitial_nucleation - kiv * Ci_eff * Cv_eff - surface_sink_strength * Di * Ci_eff
    df[1] = G0v - kiv * Ci_eff * Cv_eff - surface_sink_strength * Dv * Cv_eff

    # Stored interstitial content in faulted/perfect loops. Coalescence does not
    # appear here because merging conserves the combined loop inventory.
    # geometry_factor*Rf^2*Cf = Nf by definition.
    df[2] = faulted_absorption + 2.0 * diinterstitial_nucleation - Puf * geometry_factor * Rf**2 * Cf_eff
    df[3] = perfect_absorption + Puf * geometry_factor * Rf**2 * Cf_eff

    # Faulted and perfect loop number densities. Reducing C at fixed stored
    # inventory N increases the content-equivalent radius
    # R = sqrt(Omega0*N/(pi*b*C)); no separate radius ODE is required.
    df[4] = diinterstitial_nucleation - Puf * Cf_eff - faulted_coalescence_loss
    df[5] = Puf * Cf_eff - perfect_coalescence_loss

    if not np.all(np.isfinite(df)):
        raise FloatingPointError("Non-finite ODE derivative.")

    return df
