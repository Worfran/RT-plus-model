"""Plot every TEM image as an independent number-density dataset.

The script creates one figure for each temperature in the selected specimen
series.  BF and DF images are placed on separate rows, and every image receives
its own panel.  Histograms use the image-specific sampled volume and a fixed
diameter-bin width:

    density_j = n_j / (V_image * Delta_D)

The plotted units are therefore nm^-4.

Run from the project root:

    python scripts/plot-experimental-images.py
    python scripts/plot-experimental-images.py --series pristine
    python scripts/plot-experimental-images.py --series all --bin-width-nm 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtplus.config import DataConfig, EVENT_SERIES
from rtplus.data_loader import load_all_loop_data
from rtplus.plotting import plot_experimental_images_per_temperature


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot volume- and bin-width-normalized loop distributions for "
            "every experimental image."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "Data",
        help="Directory containing the loop CSV files and Volume_per_image.csv.",
    )
    parser.add_argument(
        "--series",
        choices=["irradiated", "pristine", "all"],
        default="irradiated",
        help="Specimen series to plot.",
    )
    parser.add_argument(
        "--bin-width-nm",
        type=float,
        default=1.0,
        help="Fixed diameter-bin width in nm. Default: 1.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Results" / "experimental-per-image",
        help="Directory for the generated PNG files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures after saving them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bin_width_nm <= 0.0:
        raise ValueError("--bin-width-nm must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    loop_data = load_all_loop_data(
        DataConfig(data_dir=args.data_dir),
        project_root=PROJECT_ROOT,
    )
    selected_series = (
        list(EVENT_SERIES)
        if args.series == "all"
        else [args.series]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for series_id in selected_series:
        figures = plot_experimental_images_per_temperature(
            loop_data=loop_data,
            event_series=EVENT_SERIES,
            series_id=series_id,
            bin_width_nm=args.bin_width_nm,
        )
        events_by_order = {
            event.event_order: event
            for event in EVENT_SERIES[series_id]
        }
        for event_order, figure in figures.items():
            event = events_by_order[event_order]
            temperature_label = f"{event.temperature_C:g}".replace(".", "p")
            path = args.output_dir / (
                f"{series_id}-event-{event_order}-{temperature_label}C-per-image.png"
            )
            figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
            saved_paths.append(path)

    print("\nSaved per-image experimental distributions:")
    for path in saved_paths:
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
