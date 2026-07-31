"""Step 0 test: are pristine and irradiated TEM observations different?

Run from the project root with the project virtual environment:

    python scripts/step0-compare-pristine-irradiated.py
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

from rtplus.config import DataConfig
from rtplus.data_loader import load_all_loop_data, print_dataset_summary
from rtplus.series_comparison import (
    compare_matched_pristine_irradiated,
    plot_image_density_comparisons,
    plot_matched_density_spectra,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 0 image-clustered comparison of pristine and irradiated "
            "loop densities and diameter distributions."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "Data",
        help="Directory containing the loop CSV files and Volume_per_image.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Results" / "step0-pristine-vs-irradiated",
        help="Destination for the CSV report, notes, and PNG figures.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="False-discovery-rate threshold used for the interpretation column.",
    )
    parser.add_argument(
        "--focus-quantile",
        type=float,
        default=0.99,
        help="Upper pooled diameter quantile displayed in each spectrum panel.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the statistical table and notes without generating figures.",
    )
    return parser.parse_args()


def _write_method_notes(path: Path, alpha: float) -> None:
    path.write_text(
        "\n".join(
            [
                "STEP 0: PRISTINE VERSUS IRRADIATED OBSERVABLE DATA",
                "",
                "The exchangeable unit is a complete TEM image, not an individual loop.",
                "Exact tests enumerate all image-label assignments while preserving group sizes.",
                "Density statistic: absolute log ratio of mean image loop densities.",
                "Shape statistic: image-balanced 1-Wasserstein diameter distance.",
                "Joint statistic: Euclidean combination of log-density difference and",
                "Wasserstein distance normalized by the pooled image-balanced median diameter.",
                "Benjamini-Hochberg q-values adjust the four matched temperature-mode tests.",
                f"Interpretation threshold: alpha={alpha:g}.",
                "",
                "These are raw observable comparisons. No fitted TEM visibility correction is",
                "applied, preventing a group-specific correction from removing the difference",
                "being tested. With only one or two pristine images per matched group, exact",
                "p-values are coarse and failure to reject does not demonstrate equivalence.",
                "Images from the same specimen may also be technical rather than biological",
                "replicates, so causal generalization requires independently prepared specimens.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in the interval (0, 1).")
    if not 0.0 < args.focus_quantile <= 1.0:
        raise ValueError("--focus-quantile must be in the interval (0, 1].")

    loop_data = load_all_loop_data(
        DataConfig(data_dir=args.data_dir),
        project_root=PROJECT_ROOT,
    )
    print_dataset_summary(loop_data)
    results = compare_matched_pristine_irradiated(loop_data, alpha=args.alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "step0-comparison-results.csv"
    notes_path = args.output_dir / "step0-method-notes.txt"
    results.to_csv(table_path, index=False)
    _write_method_notes(notes_path, args.alpha)

    display_columns = [
        "temperature_C",
        "mode",
        "n_images_pristine",
        "n_images_irradiated",
        "density_ratio",
        "wasserstein_nm",
        "density_p_exact",
        "shape_p_exact",
        "joint_p_exact",
        "joint_q_bh",
        "joint_min_p",
        "interpretation",
    ]
    print("\nSTEP 0 IMAGE-CLUSTERED COMPARISON")
    print(results[display_columns].to_string(index=False))
    print(
        "\nCaution: exact p-values are limited by the number of independent "
        "images; non-significance is not evidence of equivalence."
    )
    print(f"\nSaved table: {table_path}")
    print(f"Saved method notes: {notes_path}")

    if not args.no_plots:
        spectra = plot_matched_density_spectra(
            loop_data,
            results,
            focus_quantile=args.focus_quantile,
        )
        density_points = plot_image_density_comparisons(loop_data, results)
        spectra_path = args.output_dir / "01-matched-density-spectra.png"
        density_path = args.output_dir / "02-image-density-comparison.png"
        spectra.savefig(spectra_path, dpi=300, bbox_inches="tight")
        density_points.savefig(density_path, dpi=300, bbox_inches="tight")
        plt.close(spectra)
        plt.close(density_points)
        print(f"Saved distribution figure: {spectra_path}")
        print(f"Saved image-density figure: {density_path}")


if __name__ == "__main__":
    main()
