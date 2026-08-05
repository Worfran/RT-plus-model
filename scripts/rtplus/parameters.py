"""Parameter-vector management.

The goal is to avoid fragile hard-coded unpacking in the main script. Adding a
new fitted parameter should mostly happen by editing DEFAULT_PARAMETER_SPECS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    scope: str  # "global" or "per_temperature"
    transform: str  # "identity" or "log"
    initial: float
    bounds: tuple[float, float]

    def to_vector(self, value: float) -> float:
        if self.transform == "identity":
            return float(value)
        if self.transform == "log":
            return float(np.log(value))
        raise ValueError(f"Unknown transform: {self.transform}")

    def from_vector(self, value: float) -> float:
        if self.transform == "identity":
            return float(value)
        if self.transform == "log":
            return float(np.exp(value))
        raise ValueError(f"Unknown transform: {self.transform}")

    def vector_bounds(self) -> tuple[float, float]:
        low, high = self.bounds
        if self.transform == "identity":
            return float(low), float(high)
        if self.transform == "log":
            return float(np.log(low)), float(np.log(high))
        raise ValueError(f"Unknown transform: {self.transform}")


KINETIC_PARAMETER_SPECS = (
    ParameterSpec("Em", scope="global", transform="identity", initial=2.8, bounds=(0.1, 6.0)),
    ParameterSpec("Ea", scope="global", transform="identity", initial=1.9, bounds=(0.1, 6.0)),
    ParameterSpec("P0", scope="global", transform="log", initial=1e-12, bounds=(1e-30, 1e-6)),
    # Effective coalescence/ripening of the faulted-loop population.  This is
    # separate from perfect-loop coalescence because the two loop types have
    # different mobility and structure.
    ParameterSpec("Ea_f", scope="global", transform="identity", initial=1.9, bounds=(0.1, 6.0)),
    ParameterSpec("P0_f", scope="global", transform="log", initial=1e-12, bounds=(1e-30, 1e-6)),
    ParameterSpec("Puf", scope="per_temperature", transform="log", initial=1e-5, bounds=(1e-10, 1e-2)),
    # Rel-rod data directly identify the faulted-loop width. The initial
    # as-irradiated population has its own width and each simulated annealing
    # temperature has another, allowing the distribution to broaden or narrow
    # as the specimen evolves. The same event-specific width is also used for
    # the faulted component of BF.
    # Bawane et al. model faulted-loop radii with an ordinary Gaussian and
    # report omega/R generally near 1.0-1.2.  Do not cap this ratio at 1/3 or
    # renormalize the negative tail: the paper's observable moments integrate
    # the Gaussian only over the detectable positive-radius range.
    ParameterSpec(
        "k_f_initial",
        scope="global",
        transform="log",
        initial=1.0,
        bounds=(0.15, 1.5),
    ),
    ParameterSpec(
        "k_f",
        scope="per_temperature",
        transform="log",
        initial=1.0,
        bounds=(0.15, 1.5),
    ),
    ParameterSpec("k_p", scope="global", transform="log", initial=0.5, bounds=(0.05, 3.0)),
)


INITIAL_CONDITION_PARAMETER_SPECS = (
    # All concentrations use cm^-3.  Nf0 and Np0 are deliberately omitted:
    # they are derived from C and R so the initial state is not redundant.
    ParameterSpec("Ci0", scope="global", transform="log", initial=1.4e17, bounds=(1e10, 1e20)),
    ParameterSpec("Cf0", scope="global", transform="log", initial=8.0e16, bounds=(1e12, 1e20)),
    # A vanishing Cp combined with an enormous Rp is a singular finite-mixture
    # solution, not a physically resolved perfect-loop population.
    ParameterSpec("Cp0", scope="global", transform="log", initial=5.0e16, bounds=(1e15, 1e20)),
    ParameterSpec("Rf0_nm", scope="global", transform="log", initial=0.62, bounds=(0.1, 20.0)),
    ParameterSpec("Rp0_nm", scope="global", transform="log", initial=2.5, bounds=(0.1, 20.0)),
)


VACANCY_EXTENSION_PARAMETER_SPECS = (
    # This optional extension is intentionally separate from the source loop
    # model. Cv0 is otherwise fixed to zero and Ev is unused.
    ParameterSpec("Cv0", scope="global", transform="log", initial=1e16, bounds=(1e8, 1e21)),
    ParameterSpec("Ev", scope="global", transform="identity", initial=0.59, bounds=(0.1, 6.0)),
)


DEFAULT_PARAMETER_SPECS = (
    KINETIC_PARAMETER_SPECS
    + INITIAL_CONDITION_PARAMETER_SPECS
)


def parameter_specs(enable_vacancy_extension: bool = False):
    specs = DEFAULT_PARAMETER_SPECS
    if enable_vacancy_extension:
        specs += VACANCY_EXTENSION_PARAMETER_SPECS
    return specs


def get_parameter_temperatures(event_series) -> list[float]:
    """Return sorted temperatures for events that are actually simulated."""
    return sorted({
        float(event.temperature_C)
        for events in event_series.values()
        for event in events
        if event.simulate and event.duration_s > 0
    })


def build_theta0_and_bounds(temperatures: Iterable[float], specs=DEFAULT_PARAMETER_SPECS):
    theta0 = []
    bounds = []
    temps = [float(T) for T in temperatures]

    for spec in specs:
        if spec.scope == "global":
            theta0.append(spec.to_vector(spec.initial))
            bounds.append(spec.vector_bounds())
        elif spec.scope == "per_temperature":
            for _ in temps:
                theta0.append(spec.to_vector(spec.initial))
                bounds.append(spec.vector_bounds())
        else:
            raise ValueError(f"Unknown scope: {spec.scope}")

    return np.asarray(theta0, dtype=float), bounds


def unpack_theta(theta_vec, temperatures: Iterable[float], specs=DEFAULT_PARAMETER_SPECS) -> dict:
    theta_vec = np.asarray(theta_vec, dtype=float)
    temps = [float(T) for T in temperatures]
    theta = {}
    idx = 0

    for spec in specs:
        if spec.scope == "global":
            theta[spec.name] = spec.from_vector(theta_vec[idx])
            idx += 1
        elif spec.scope == "per_temperature":
            theta[f"{spec.name}_by_T"] = {}
            for T in temps:
                theta[f"{spec.name}_by_T"][T] = spec.from_vector(theta_vec[idx])
                idx += 1
        else:
            raise ValueError(f"Unknown scope: {spec.scope}")

    if idx != len(theta_vec):
        raise ValueError(f"Theta length mismatch. Used {idx}, got {len(theta_vec)}")

    return theta


def faulted_width_at_temperature(
    theta: dict,
    temperature_C: float | None = None,
) -> float:
    """Return the faulted-loop coefficient of variation for one event.

    ``k_f_initial`` applies to the as-irradiated observation. Annealed events
    use ``k_f_by_T``. The legacy global ``k_f`` form remains accepted so that
    lower-level helpers and old diagnostic inputs stay usable.
    """

    widths_by_temperature = theta.get("k_f_by_T", {})
    if temperature_C is not None:
        requested_temperature = float(temperature_C)
        for stored_temperature, value in widths_by_temperature.items():
            if np.isclose(
                float(stored_temperature),
                requested_temperature,
                rtol=0.0,
                atol=1.0e-9,
            ):
                return float(value)

        if widths_by_temperature:
            raise KeyError(
                "Missing faulted-loop width for "
                f"T={requested_temperature:g} C."
            )

    if temperature_C is None and "k_f_initial" in theta:
        return float(theta["k_f_initial"])
    if "k_f" in theta:
        return float(theta["k_f"])

    if temperature_C is None and len(widths_by_temperature) == 1:
        return float(next(iter(widths_by_temperature.values())))

    raise KeyError(
        "Missing faulted-loop width: expected k_f_initial, k_f, or a "
        "matching entry in k_f_by_T."
    )


def randomize_theta0(theta0, bounds, rng: np.random.Generator, scale: float = 0.35):
    """Generate a randomized optimizer initial guess inside bounds."""
    theta0 = np.asarray(theta0, dtype=float)
    out = theta0.copy()
    for i, (low, high) in enumerate(bounds):
        width = high - low
        proposal = theta0[i] + rng.normal(0.0, scale * width)
        out[i] = np.clip(proposal, low, high)
    return out
