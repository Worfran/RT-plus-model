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
    volume_metadata: pd.DataFrame,
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

    dataset_volumes = volume_metadata[
        volume_metadata["source_file"] == spec.filename
    ].copy()
    expected_images = {str(column).strip() for column in raw.columns}
    available_images = set(dataset_volumes["image"])
    missing_images = sorted(expected_images - available_images)
    if missing_images:
        raise ValueError(
            f"Volume metadata are missing for {spec.filename}: "
            + ", ".join(missing_images)
        )

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
    long_data["image"] = long_data["image"].astype(str).str.strip()
    long_data = long_data.merge(
        dataset_volumes,
        on=["source_file", "image"],
        how="left",
        validate="many_to_one",
    )

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
            "volume_nm3_reference",
            "volume_nm3_effective",
            "volume_cm3",
        ]
    ]


def load_volume_metadata(
    data_dir: Path,
    config: DataConfig,
) -> pd.DataFrame:
    """Load image volumes and convert them to the selected lamella thickness."""

    path = Path(data_dir) / config.volume_filename
    if not path.exists():
        raise FileNotFoundError(f"Image-volume metadata not found:\n{path}")

    metadata = pd.read_csv(path)
    required = {"Source File", "Image ID", "Volume (nm3)"}
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise ValueError(
            f"Volume metadata file {path} is missing columns: "
            + ", ".join(missing_columns)
        )

    metadata = metadata.rename(
        columns={
            "Source File": "source_file",
            "Image ID": "image",
            "Volume (nm3)": "volume_nm3_reference",
        }
    )[["source_file", "image", "volume_nm3_reference"]].copy()
    metadata["source_file"] = metadata["source_file"].astype(str).str.strip()
    metadata["image"] = metadata["image"].astype(str).str.strip()
    metadata["volume_nm3_reference"] = pd.to_numeric(
        metadata["volume_nm3_reference"],
        errors="coerce",
    )

    invalid = (
        metadata["source_file"].eq("")
        | metadata["image"].eq("")
        | ~np.isfinite(metadata["volume_nm3_reference"])
        | (metadata["volume_nm3_reference"] <= 0.0)
    )
    if invalid.any():
        raise ValueError(f"Invalid row(s) in image-volume metadata: {path}")

    duplicate_keys = metadata.duplicated(
        subset=["source_file", "image"],
        keep=False,
    )
    if duplicate_keys.any():
        duplicates = metadata.loc[
            duplicate_keys,
            ["source_file", "image"],
        ].drop_duplicates()
        raise ValueError(
            "Duplicate source-file/image keys in volume metadata:\n"
            + duplicates.to_string(index=False)
        )

    if (
        config.volume_reference_thickness_nm <= 0.0
        or config.analysis_thickness_nm <= 0.0
    ):
        raise ValueError("Lamella thicknesses must be positive.")

    thickness_scale = (
        float(config.analysis_thickness_nm)
        / float(config.volume_reference_thickness_nm)
    )
    metadata["volume_nm3_effective"] = (
        metadata["volume_nm3_reference"] * thickness_scale
    )
    metadata["volume_cm3"] = metadata["volume_nm3_effective"] * 1.0e-21
    return metadata


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
    volume_metadata = load_volume_metadata(data_dir, config)

    datasets = [
        load_one_dataset(
            data_dir=data_dir,
            spec=spec,
            volume_metadata=volume_metadata,
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
