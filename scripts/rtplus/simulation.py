"""Temperature simulation wrapper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import MaterialConstants, SimulationConfig
from .ode import rhs
from .physics import coalescence_rate, compute_radius, diffusion_coeff, compute_radius


@dataclass(frozen=True)
class Prediction:
    temperature_C: float
    y_final: np.ndarray
    Rf: float
    Rp: float
    Cf: float
    Cp: float
    Di: float
    Pcs: float
    Pfcs: float
    Puf: float


def simulate_temperature(
    T_C: float,
    theta: dict,
    material: MaterialConstants,
    sim_config: SimulationConfig,
    y0: np.ndarray,
) -> Prediction:
    T_C = float(T_C)
    T_K = T_C + 273.15

    Puf_by_T = theta.get("Puf_by_T")
    if Puf_by_T is None or T_C not in Puf_by_T:
        raise KeyError(f"Missing Puf value for T={T_C}")

    params = {
        "a": material.a,
        "b": material.b,
        "Omega0": material.Omega0,
        "r0": material.r0,
        "Rii": material.Rii,

        "Ziv_iK": material.Ziv_iK,
        "Ziv_vK": material.Ziv_vK,

        "G0i": material.G0i,
        "G0v": material.G0v,

        "Di": diffusion_coeff(material.D0, theta["Em"], T_K),
        "Dv": diffusion_coeff(material.Dv0, theta.get("Ev", material.Ev), T_K)
        if material.enable_vacancy_extension else 0.0,
        "enable_vacancy_extension": material.enable_vacancy_extension,
        "enable_surface_sink": material.enable_surface_sink,
        "lamella_thickness_cm": material.lamella_thickness_cm,

        "Pcs": coalescence_rate(theta["P0"], theta["Ea"], T_K),
        "Pfcs": coalescence_rate(theta["P0_f"], theta["Ea_f"], T_K),
        "Puf": Puf_by_T[T_C],
        "coalescence_model": theta.get(
            "coalescence_model",
            "interaction_driven",
        ),
    }

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(0.0, sim_config.t_end_s),
        y0=np.asarray(y0, dtype=float),
        method=sim_config.method,
        rtol=sim_config.rtol,
        atol=sim_config.atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    y_final = np.asarray(sol.y[:, -1], dtype=float)
    if not np.all(np.isfinite(y_final)):
        raise RuntimeError("Non-finite final state.")

    Ci, Cv, Nf, Np, Cf, Cp = y_final
    if Nf <= 0 or Np <= 0 or Cf <= 0 or Cp <= 0:
        raise RuntimeError(
            "Nonphysical final loop state: "
            f"Nf={Nf:.3e}, Np={Np:.3e}, Cf={Cf:.3e}, Cp={Cp:.3e}"
        )

    Rf = compute_radius(Nf, Cf, material.b, material.Omega0)
    Rp = compute_radius(Np, Cp, material.b, material.Omega0)

    return Prediction(
        temperature_C=T_C,
        y_final=y_final,
        Rf=Rf,
        Rp=Rp,
        Cf=float(Cf),
        Cp=float(Cp),
        Di=params["Di"],
        Pcs=params["Pcs"],
        Pfcs=params["Pfcs"],
        Puf=params["Puf"],
    )


def simulate_all_temperatures(temperatures, theta, material: MaterialConstants, sim_config: SimulationConfig, y0: np.ndarray):
    predictions = {}
    for T_C in temperatures:
        predictions[float(T_C)] = simulate_temperature(float(T_C), theta, material, sim_config, y0)
    return predictions

def prediction_from_state(
    y,
    material,
    temperature_C,
    event_order,
    series_id,
    metadata=None,
):
    """
    Convert one six-variable state into model observables.
    """

    y = np.asarray(y, dtype=float)

    if y.shape != (6,):
        raise ValueError(
            f"Expected state with shape (6,), received {y.shape}"
        )

    Ci, Cv, Nf, Np, Cf, Cp = y

    Rf = compute_radius(
        N=Nf,
        C=Cf,
        b=material.b,
        Omega0=material.Omega0,
    )

    Rp = compute_radius(
        N=Np,
        C=Cp,
        b=material.b,
        Omega0=material.Omega0,
    )

    return {
        "series_id": series_id,
        "event_order": int(event_order),
        "temperature_C": float(temperature_C),
        "y_initial": None,
        "y_final": y.copy(),
        "Ci": Ci,
        "Cv": Cv,
        "Nf": Nf,
        "Np": Np,
        "Cf": Cf,
        "Cp": Cp,
        "Rf": Rf,
        "Rp": Rp,
        "metadata": metadata or {},
    }

def simulate_event(
    event,
    theta,
    material,
    y0,
    series_id,
):
    """
    Simulate one event in a thermal series.
    """

    T_C = float(event.temperature_C)
    T_K = T_C + 273.15

    y0 = np.asarray(y0, dtype=float).copy()

    if not event.simulate or event.duration_s <= 0:
        result = prediction_from_state(
            y=y0,
            material=material,
            temperature_C=T_C,
            event_order=event.event_order,
            series_id=series_id,
            metadata={
                "simulated": False,
                "duration_s": 0.0,
            },
        )

        result["y_initial"] = y0.copy()
        return result

    Puf = theta["Puf_by_T"][T_C]

    params = {
        "a": material.a,
        "b": material.b,
        "Omega0": material.Omega0,
        "r0": material.r0,
        "Rii": material.Rii,

        "Ziv_iK": material.Ziv_iK,
        "Ziv_vK": material.Ziv_vK,

        "G0i": material.G0i,
        "G0v": material.G0v,

        "Di": diffusion_coeff(
            material.D0,
            theta["Em"],
            T_K,
        ),

        "Dv": diffusion_coeff(material.Dv0, theta.get("Ev", material.Ev), T_K)
        if material.enable_vacancy_extension else 0.0,
        "enable_vacancy_extension": material.enable_vacancy_extension,
        "enable_surface_sink": material.enable_surface_sink,
        "lamella_thickness_cm": material.lamella_thickness_cm,

        "Puf": Puf,

        "Pcs": coalescence_rate(
            theta["P0"],
            theta["Ea"],
            T_K,
        ),
        "Pfcs": coalescence_rate(
            theta["P0_f"],
            theta["Ea_f"],
            T_K,
        ),
        "coalescence_model": theta.get(
            "coalescence_model",
            "interaction_driven",
        ),
    }

    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(0.0, float(event.duration_s)),
        y0=y0,
        method="BDF",
        rtol=1e-6,
        atol=1e-12,
    )

    if not solution.success:
        raise RuntimeError(
            f"Integration failed for {series_id}, "
            f"event {event.event_order}, T={T_C:g} C: "
            f"{solution.message}"
        )

    y_final = np.asarray(solution.y[:, -1], dtype=float)

    if not np.all(np.isfinite(y_final)):
        raise RuntimeError(
            f"Non-finite state for {series_id}, "
            f"event {event.event_order}"
        )

    negative_tolerance = np.maximum(1e-8, 1e-10 * np.maximum(np.abs(y0), 1.0))
    if np.any(y_final < -negative_tolerance):
        raise RuntimeError(
            f"Nonphysical negative state for {series_id}, event "
            f"{event.event_order}: {y_final}"
        )
    y_final = np.maximum(y_final, 0.0)

    prediction = prediction_from_state(
        y=y_final,
        material=material,
        temperature_C=T_C,
        event_order=event.event_order,
        series_id=series_id,
        metadata={
            "simulated": True,
            "duration_s": float(event.duration_s),
            "Di": params["Di"],
            "Dv": params["Dv"],
            "Puf": params["Puf"],
            "Pcs": params["Pcs"],
            "Pfcs": params["Pfcs"],
        },
    )

    prediction["y_initial"] = y0.copy()

    return prediction

def simulate_series(
    series_id,
    events,
    theta,
    material,
    initial_state,
):
    """
    Simulate one complete experimental history.

    The final state of event j becomes the initial state of event j+1.
    """

    ordered_events = sorted(
        events,
        key=lambda event: event.event_order,
    )

    current_state = np.asarray(
        initial_state,
        dtype=float,
    ).copy()

    predictions = {}

    for event in ordered_events:
        prediction = simulate_event(
            event=event,
            theta=theta,
            material=material,
            y0=current_state,
            series_id=series_id,
        )

        predictions[event.event_order] = prediction

        current_state = prediction["y_final"].copy()

    return predictions

def simulate_all_series(
    event_series,
    theta,
    material,
    initial_states,
):
    """
    Simulate every experimental series sequentially.

    Different series are independent, but events inside one series
    are always sequential.
    """

    all_predictions = {}

    for series_id, events in event_series.items():
        if series_id not in initial_states:
            raise KeyError(
                f"No initial state provided for series '{series_id}'"
            )

        all_predictions[series_id] = simulate_series(
            series_id=series_id,
            events=events,
            theta=theta,
            material=material,
            initial_state=initial_states[series_id],
        )

    return all_predictions
