"""Configuration objects for the local RT+ fitting script.

This file contains fixed choices only. Change defaults here when you want a
project-wide change, not inside the ODE/objective/plotting functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MaterialConstants:
    """Material and geometric constants.

    Units:
        length: cm
        volume: cm^3
        loop density: cm^-3

    Convention used here:
        The Frank-loop 1/3 factor is included inside b.

        b = |a/3 <111>| = a*sqrt(3)/3

        Therefore:
            N = (pi*b/Omega0)*R^2*C
            R = sqrt(Omega0*N/(pi*b*C))
    """

    a: float = 5.41e-8
    b: float = field(default_factory=lambda: 5.41e-8 * np.sqrt(3.0) / 3.0)
    Omega0: float = field(default_factory=lambda: (5.41e-8 ** 3) / 4.0)
    r0: float = field(default_factory=lambda: 2.0 * 5.41e-8)
    Rii: float = field(default_factory=lambda: (np.sqrt(3.0) / 2.0) * 5.41e-8)
    Zii: float = 12.0
    # Interstitial-vacancy recombination bias/capture factors
    Ziv_iK: float = 48.0
    Ziv_vK: float = 48.0

    # Vacancy diffusion prefactor, cm^2/s
    # Keep zero if vacancy diffusion is inactive for now.
    Dv0: float = 0.0

    # Paper value 1e6 um^2/s converted to cm^2/s: 1 um^2 = 1e-8 cm^2.
    D0: float = 1.0e-2

    # Optional point-defect source terms. Keep zero unless you explicitly add irradiation generation.
    G0i: float = 0.0
    G0v: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    t_end_s: float = 3600.0
    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-12


@dataclass(frozen=True)
class FitConfig:
    """Fit and optimization settings.

    Parallelization rule:
        Only optimizer starts are parallelized. Inside each objective call,
        temperatures are solved sequentially.
    """

    fit_temperatures: Optional[Sequence[float]] = None
    exclude_room_temperature: bool = False
    n_starts: int = 1
    parallel_starts: bool = False
    max_workers: Optional[int] = None
    random_seed: int = 10
    maxiter: int = 2000
    ftol: float = 1e-9
    gtol: float = 1e-6
    objective_fail_value: float = 1e100

@dataclass(frozen=True)
class DatasetSpec:
    """Metadata associated with one experimental CSV file."""

    filename: str
    temperature_C: float
    mode: str
    irradiated: bool
    series_id: str
    event_order: int

@dataclass(frozen=True)
class DataConfig:
    """Data file names and their experimental-series metadata."""

    data_dir: Path = Path("Data")

    dataset_specs: tuple[DatasetSpec, ...] = (
        # Irradiated series: RT -> 900 C -> 1100 C
        DatasetSpec(
            filename="RT-BF-irr.csv",
            temperature_C=25.0,
            mode="BF",
            irradiated=True,
            series_id="irradiated",
            event_order=0,
        ),
        DatasetSpec(
            filename="RT-DF-irr.csv",
            temperature_C=25.0,
            mode="DF",
            irradiated=True,
            series_id="irradiated",
            event_order=0,
        ),
        DatasetSpec(
            filename="900-BF-irr.csv",
            temperature_C=900.0,
            mode="BF",
            irradiated=True,
            series_id="irradiated",
            event_order=1,
        ),
        DatasetSpec(
            filename="900-DF-irr.csv",
            temperature_C=900.0,
            mode="DF",
            irradiated=True,
            series_id="irradiated",
            event_order=1,
        ),
        DatasetSpec(
            filename="1100-BF-irr.csv",
            temperature_C=1100.0,
            mode="BF",
            irradiated=True,
            series_id="irradiated",
            event_order=2,
        ),
        DatasetSpec(
            filename="1100-DF-irr.csv",
            temperature_C=1100.0,
            mode="DF",
            irradiated=True,
            series_id="irradiated",
            event_order=2,
        ),

        # Pristine series: RT -> 900 C
        DatasetSpec(
            filename="RT-BF.csv",
            temperature_C=25.0,
            mode="BF",
            irradiated=False,
            series_id="pristine",
            event_order=0,
        ),
        DatasetSpec(
            filename="RT-DF.csv",
            temperature_C=25.0,
            mode="DF",
            irradiated=False,
            series_id="pristine",
            event_order=0,
        ),
        DatasetSpec(
            filename="900-BF.csv",
            temperature_C=900.0,
            mode="BF",
            irradiated=False,
            series_id="pristine",
            event_order=1,
        ),
        DatasetSpec(
            filename="900-DF.csv",
            temperature_C=900.0,
            mode="DF",
            irradiated=False,
            series_id="pristine",
            event_order=1,
        ),
    )

@dataclass(frozen=True)
class EventConfig:
    """
    One experimental event in an ordered thermal history.

    event_order:
        Position in the sequence.

    temperature_C:
        Temperature associated with the experimental dataset.

    duration_s:
        Duration of the simulation at this temperature.

    simulate:
        False means that the event is an observation of the initial state.
        True means that the ODE is integrated for duration_s.
    """

    event_order: int
    temperature_C: float
    duration_s: float
    simulate: bool = True


EVENT_SERIES = {
    "irradiated": [
        EventConfig(
            event_order=0,
            temperature_C=25.0,
            duration_s=0.0,
            simulate=False,
        ),
        EventConfig(
            event_order=1,
            temperature_C=900.0,
            duration_s=3600.0,
            simulate=True,
        ),
        EventConfig(
            event_order=2,
            temperature_C=1100.0,
            duration_s=3600.0,
            simulate=True,
        ),
    ],

    "pristine": [
        EventConfig(
            event_order=0,
            temperature_C=25.0,
            duration_s=0.0,
            simulate=False,
        ),
        EventConfig(
            event_order=1,
            temperature_C=900.0,
            duration_s=3600.0,
            simulate=True,
        ),
    ],
}
