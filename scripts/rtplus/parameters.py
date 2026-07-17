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


DEFAULT_PARAMETER_SPECS = (
    ParameterSpec("Em", scope="global", transform="identity", initial=2.8, bounds=(0.1, 6.0)),
    ParameterSpec("Ea", scope="global", transform="identity", initial=1.9, bounds=(0.1, 6.0)),
    ParameterSpec("P0", scope="global", transform="log", initial=1e-12, bounds=(1e-30, 1e-6)),
    ParameterSpec("Puf", scope="per_temperature", transform="log", initial=1e-5, bounds=(1e-10, 1e-2)),
    ParameterSpec("k_f", scope="global", transform="log", initial=0.5, bounds=(0.05, 3.0)),
    ParameterSpec("k_p", scope="global", transform="log", initial=0.5, bounds=(0.05, 3.0)),
)


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


def randomize_theta0(theta0, bounds, rng: np.random.Generator, scale: float = 0.35):
    """Generate a randomized optimizer initial guess inside bounds."""
    theta0 = np.asarray(theta0, dtype=float)
    out = theta0.copy()
    for i, (low, high) in enumerate(bounds):
        width = high - low
        proposal = theta0[i] + rng.normal(0.0, scale * width)
        out[i] = np.clip(proposal, low, high)
    return out
