"""Configuration objects for the local RT+ fitting script.

This file contains fixed choices only. Change defaults here when you want a
project-wide change, not inside the ODE/objective/plotting functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

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
    # Interstitial-vacancy recombination bias/capture factors
    Ziv_iK: float = 48.0
    Ziv_vK: float = 48.0

    # Ceria oxygen-vacancy diffusion prefactor, cm^2/s.  The value is the
    # 120e6 um^2/s prefactor cited in Table 4 of Bawane et al.  Vacancy
    # coupling is disabled by default because it is an extension of the
    # paper's five-variable loop model, not part of equations S1-S5.
    Dv0: float = 1.2
    Ev: float = 0.59
    enable_vacancy_extension: bool = False

    # The paper's loop model does not include a free-surface sink.  Retain it
    # as an explicit optional extension for a 100 nm TEM lamella.
    enable_surface_sink: bool = False
    lamella_thickness_cm: float = 1.0e-5

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
    density_loss_weight: float = 1.0
    density_relative_uncertainty_floor: float = 0.20
    # Default objective: equal weight per TEM image for loop diameters plus
    # an overdispersed count term using that image's sampled volume.
    objective_mode: str = "image_balanced_extended"
    # NB2 alpha floor.  alpha=0.04 gives a 20% asymptotic count CV.
    count_overdispersion_floor: float = 0.04
    # The faulted-loop family is deliberately Gaussian.  Retain the complete
    # small-loop side and trim only the largest skewed tail from the size loss.
    # Counts still use every observed loop.
    faulted_size_fit_fraction: float = 0.95
    # Events listed here retain the complete DF diameter distribution.  The
    # 1100 C data are broad but not treated as outliers because their upper tail
    # is part of the annealing response that the ODE must reproduce.
    faulted_full_distribution_temperatures: Sequence[float] = (1100.0,)
    # The as-irradiated observation defines the initial condition inherited by
    # every annealing step.  Give that boundary-condition measurement extra
    # weight without freezing it or fitting it in a separate stage.
    room_temperature_loss_weight: float = 3.0
    # Positive size-distribution family used for faulted loops in both DF and
    # the faulted contribution to BF.
    faulted_distribution: str = "normal"
    # Smooth TEM visibility is treated as a calibrated observation setting,
    # not inferred from the same histograms whose physical width is being fit.
    # Fitting both simultaneously is poorly identifiable.
    apply_smooth_visibility: bool = True
    Rvis_DF_nm: float = 0.50
    dRvis_DF_nm: float = 0.15
    Rvis_BF_nm: float = 1.00
    dRvis_BF_nm: float = 0.25
    # Relative image thresholds are calibrated before the physical fit from
    # same-event, same-mode size contrasts.  The offsets are centered within
    # each comparison group and frozen during RT+ optimization.
    image_specific_visibility: bool = True
    visibility_offset_sd_nm: float = 0.20
    visibility_max_offset_nm: float = 0.50
    image_visibility_rvis_nm: Mapping[tuple, float] = field(default_factory=dict)
    image_visibility_offsets_nm: Mapping[tuple, float] = field(default_factory=dict)
    image_visibility_efficiency: Mapping[tuple, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationConfig:
    """TEM visibility and resolution model from Bawane et al., Eqs. 2-6.

    The fitted smooth visibility parameters are radii, matching the physical
    visibility function w(R). The older hard resolution limits remain
    available only for explicit legacy checks.
    """

    relrod_faulted_visibility: float = 0.25
    bf_faulted_visibility: float = 1.0
    bf_perfect_visibility: float = 0.5
    relrod_resolution_radius_nm: float = 0.5
    bf_resolution_radius_nm: float = 1.0
    # The current raw files contain many measured diameters below twice the
    # published radius limits, so applying those limits would assign zero
    # probability to recorded observations.  Keep the correction available
    # for a genuinely thresholded dataset.
    apply_resolution_cutoff: bool = False


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
    volume_filename: str = "Volume_per_image.csv"
    volume_reference_thickness_nm: float = 80.0
    analysis_thickness_nm: float = 100.0

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
