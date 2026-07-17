"""Load and organize experimental BF/DF loop-size datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, DatasetSpec, EVENT_SERIES


def resolve_data_directory(
    config: DataConfig,
    project_root: Path | None = None,
) -> Path:
    """
    Resolve the directory containing the CSV files.

    Relative paths are interpreted relative to project_root when supplied.
    Otherwise, they are interpreted relative to the current working directory.
    """

    data_dir = Path(config.data_dir)

    if data_dir.is_absolute():
        return data_dir

    if project_root is not None:
        return Path(project_root) / data_dir

    return Path.cwd() / data_dir


def load_one_dataset(
    data_dir: Path,
    spec: DatasetSpec,
) -> pd.DataFrame:
    """
    Load one wide CSV file and convert it into long format.

    Every finite positive value is treated as one measured loop diameter.
    """

    file_path = Path(data_dir) / spec.filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Experimental dataset not found:\n{file_path}"
        )

    raw = pd.read_csv(file_path)

    if raw.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    # Convert:
    #
    # Image 1 | Image 2 | Image 3
    #
    # into:
    #
    # image   size
    # Image 1 value
    # Image 2 value
    # ...
    long_data = raw.melt(
        var_name="image",
        value_name="size",
    )

    long_data["size"] = pd.to_numeric(
        long_data["size"],
        errors="coerce",
    )

    long_data = long_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    long_data = long_data.dropna(
        subset=["size"]
    ).copy()

    long_data = long_data[
        long_data["size"] > 0
    ].copy()

    long_data["temperature_C"] = float(spec.temperature_C)
    long_data["mode"] = spec.mode.upper()
    long_data["irradiated"] = bool(spec.irradiated)
    long_data["series_id"] = spec.series_id
    long_data["event_order"] = int(spec.event_order)
    long_data["source_file"] = spec.filename

    return long_data[
        [
            "size",
            "temperature_C",
            "mode",
            "irradiated",
            "series_id",
            "event_order",
            "image",
            "source_file",
        ]
    ]


def validate_dataset_specs(
    config: DataConfig,
) -> None:
    """
    Check that dataset metadata agree with EVENT_SERIES.
    """

    for spec in config.dataset_specs:
        if spec.series_id not in EVENT_SERIES:
            raise ValueError(
                f"Dataset '{spec.filename}' refers to unknown series "
                f"'{spec.series_id}'."
            )

        matching_events = [
            event
            for event in EVENT_SERIES[spec.series_id]
            if event.event_order == spec.event_order
        ]

        if len(matching_events) != 1:
            raise ValueError(
                f"Dataset '{spec.filename}' refers to event "
                f"{spec.event_order}, but that event is not uniquely defined "
                f"for series '{spec.series_id}'."
            )

        event = matching_events[0]

        if not np.isclose(
            spec.temperature_C,
            event.temperature_C,
        ):
            raise ValueError(
                f"Temperature mismatch for '{spec.filename}': "
                f"dataset specification gives {spec.temperature_C:g} C, "
                f"while EVENT_SERIES gives {event.temperature_C:g} C."
            )

        if spec.mode.upper() not in {"BF", "DF"}:
            raise ValueError(
                f"Invalid observation mode for '{spec.filename}': "
                f"{spec.mode!r}"
            )


def load_all_loop_data(
    config: DataConfig,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Load all configured CSV files into one long dataframe.
    """

    validate_dataset_specs(config)

    data_dir = resolve_data_directory(
        config=config,
        project_root=project_root,
    )

    datasets = [
        load_one_dataset(
            data_dir=data_dir,
            spec=spec,
        )
        for spec in config.dataset_specs
    ]

    if not datasets:
        raise ValueError("No datasets were configured.")

    loop_data = pd.concat(
        datasets,
        ignore_index=True,
    )

    loop_data = loop_data.sort_values(
        by=[
            "series_id",
            "event_order",
            "mode",
            "source_file",
            "image",
        ],
    ).reset_index(drop=True)

    return loop_data


def print_dataset_summary(
    loop_data: pd.DataFrame,
) -> None:
    """Print the number of measured loops for every event and mode."""

    summary = (
        loop_data
        .groupby(
            [
                "series_id",
                "event_order",
                "temperature_C",
                "mode",
            ],
            sort=True,
        )
        .size()
        .rename("n_loops")
    )

    print("\nLoaded experimental event series:")
    print(summary)
    print(f"\nTotal measured loops: {len(loop_data)}")