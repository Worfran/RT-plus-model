"""Image-clustered Step 0 comparison of pristine and irradiated TEM data."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import energy_distance, wasserstein_distance


@dataclass(frozen=True)
class ImageSample:
    """One TEM image, the exchangeable unit used by the exact tests."""

    series_id: str
    source_file: str
    image_id: str
    volume_nm3: float
    diameters_nm: np.ndarray

    @property
    def loop_count(self) -> int:
        return int(self.diameters_nm.size)

    @property
    def density_nm3(self) -> float:
        return float(self.loop_count / self.volume_nm3)


def _extract_image_samples(group: pd.DataFrame, series_id: str) -> list[ImageSample]:
    selected = group[group["series_id"] == series_id]
    samples: list[ImageSample] = []
    for (source_file, image_id), image_data in selected.groupby(
        ["source_file", "image"],
        sort=True,
    ):
        volumes = image_data["volume_nm3_effective"].drop_duplicates().to_numpy(float)
        if volumes.size != 1 or not np.isfinite(volumes[0]) or volumes[0] <= 0.0:
            raise ValueError(
                f"Image {source_file}/{image_id} must have one positive volume."
            )
        diameters = image_data["size"].to_numpy(dtype=float)
        diameters = diameters[np.isfinite(diameters) & (diameters > 0.0)]
        if diameters.size == 0:
            raise ValueError(f"Image {source_file}/{image_id} has no loop diameters.")
        samples.append(
            ImageSample(
                series_id=str(series_id),
                source_file=str(source_file),
                image_id=str(image_id),
                volume_nm3=float(volumes[0]),
                diameters_nm=diameters,
            )
        )
    if not samples:
        raise ValueError(f"No images found for series {series_id!r}.")
    return samples


def _equal_image_empirical_distribution(
    samples: list[ImageSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool diameters while assigning equal total weight to every image."""

    if not samples:
        raise ValueError("At least one image is required.")
    values = np.concatenate([sample.diameters_nm for sample in samples])
    weights = np.concatenate(
        [
            np.full(
                sample.loop_count,
                1.0 / (len(samples) * sample.loop_count),
                dtype=float,
            )
            for sample in samples
        ]
    )
    return values, weights


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values, dtype=float)[order]
    ordered_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights) / np.sum(ordered_weights)
    return float(ordered_values[np.searchsorted(cumulative, 0.5, side="left")])


def _partition_statistics(
    pristine: list[ImageSample],
    irradiated: list[ImageSample],
) -> dict[str, float]:
    density_pristine = float(np.mean([sample.density_nm3 for sample in pristine]))
    density_irradiated = float(np.mean([sample.density_nm3 for sample in irradiated]))
    if density_pristine <= 0.0 or density_irradiated <= 0.0:
        raise ValueError("Image number densities must be positive.")

    pristine_values, pristine_weights = _equal_image_empirical_distribution(pristine)
    irradiated_values, irradiated_weights = _equal_image_empirical_distribution(
        irradiated
    )
    density_ratio = density_irradiated / density_pristine
    log_density_difference = abs(float(np.log(density_ratio)))
    wasserstein_nm = float(
        wasserstein_distance(
            pristine_values,
            irradiated_values,
            u_weights=pristine_weights,
            v_weights=irradiated_weights,
        )
    )
    energy_statistic = float(
        energy_distance(
            pristine_values,
            irradiated_values,
            u_weights=pristine_weights,
            v_weights=irradiated_weights,
        )
    )
    pooled_values = np.concatenate([pristine_values, irradiated_values])
    pooled_weights = np.concatenate(
        [0.5 * pristine_weights, 0.5 * irradiated_weights]
    )
    diameter_scale_nm = max(_weighted_median(pooled_values, pooled_weights), 1.0e-12)
    normalized_shape_difference = wasserstein_nm / diameter_scale_nm
    joint_discrepancy = float(
        np.hypot(log_density_difference, normalized_shape_difference)
    )
    mean_pristine_nm = float(np.sum(pristine_values * pristine_weights))
    mean_irradiated_nm = float(np.sum(irradiated_values * irradiated_weights))
    return {
        "density_pristine_nm3": density_pristine,
        "density_irradiated_nm3": density_irradiated,
        "density_ratio": density_ratio,
        "density_statistic": log_density_difference,
        "mean_diameter_pristine_nm": mean_pristine_nm,
        "mean_diameter_irradiated_nm": mean_irradiated_nm,
        "mean_diameter_shift_nm": mean_irradiated_nm - mean_pristine_nm,
        "wasserstein_nm": wasserstein_nm,
        "energy_statistic": energy_statistic,
        "normalized_shape_statistic": normalized_shape_difference,
        "joint_statistic": joint_discrepancy,
    }


def _exact_upper_tail_pvalue(values: np.ndarray, observed: float) -> float:
    tolerance = 1.0e-12 * max(1.0, abs(float(observed)))
    return float(np.mean(values >= float(observed) - tolerance))


def _minimum_attainable_pvalue(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    tolerance = 1.0e-12 * max(1.0, abs(maximum))
    return float(np.mean(values >= maximum - tolerance))


def compare_image_samples_exact(
    pristine: list[ImageSample],
    irradiated: list[ImageSample],
) -> dict[str, float | int]:
    """Compare two series by exhaustively permuting complete TEM images."""

    observed = _partition_statistics(pristine, irradiated)
    all_samples = list(pristine) + list(irradiated)
    n_pristine = len(pristine)
    n_assignments = comb(len(all_samples), n_pristine)
    permutation_statistics = {
        "density": [],
        "shape": [],
        "joint": [],
    }
    all_indices = set(range(len(all_samples)))
    for pristine_indices_tuple in combinations(range(len(all_samples)), n_pristine):
        pristine_indices = set(pristine_indices_tuple)
        permuted_pristine = [all_samples[index] for index in sorted(pristine_indices)]
        permuted_irradiated = [
            all_samples[index] for index in sorted(all_indices - pristine_indices)
        ]
        statistics = _partition_statistics(
            permuted_pristine,
            permuted_irradiated,
        )
        permutation_statistics["density"].append(statistics["density_statistic"])
        permutation_statistics["shape"].append(statistics["wasserstein_nm"])
        permutation_statistics["joint"].append(statistics["joint_statistic"])

    density_null = np.asarray(permutation_statistics["density"], dtype=float)
    shape_null = np.asarray(permutation_statistics["shape"], dtype=float)
    joint_null = np.asarray(permutation_statistics["joint"], dtype=float)
    return {
        **observed,
        "n_assignments": int(n_assignments),
        "density_p_exact": _exact_upper_tail_pvalue(
            density_null,
            observed["density_statistic"],
        ),
        "shape_p_exact": _exact_upper_tail_pvalue(
            shape_null,
            observed["wasserstein_nm"],
        ),
        "joint_p_exact": _exact_upper_tail_pvalue(
            joint_null,
            observed["joint_statistic"],
        ),
        "density_min_p": _minimum_attainable_pvalue(density_null),
        "shape_min_p": _minimum_attainable_pvalue(shape_null),
        "joint_min_p": _minimum_attainable_pvalue(joint_null),
    }


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""

    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or np.any(~np.isfinite(pvalues)):
        raise ValueError("pvalues must be a finite one-dimensional array.")
    if np.any((pvalues < 0.0) | (pvalues > 1.0)):
        raise ValueError("pvalues must lie in [0, 1].")
    if pvalues.size == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted_ranked = ranked * pvalues.size / np.arange(1, pvalues.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def compare_matched_pristine_irradiated(
    loop_data: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run image-level comparisons for every matched temperature and mode."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in the interval (0, 1).")
    available = {
        str(series_id): set(
            zip(
                series_data["temperature_C"].astype(float),
                series_data["mode"].astype(str).str.upper(),
            )
        )
        for series_id, series_data in loop_data.groupby("series_id")
    }
    matched = sorted(
        available.get("pristine", set()) & available.get("irradiated", set()),
        key=lambda item: (item[0], {"DF": 0, "BF": 1}.get(item[1], 2)),
    )
    if not matched:
        raise ValueError("No matched pristine/irradiated temperature-mode groups found.")

    rows = []
    for temperature_C, mode in matched:
        group = loop_data[
            np.isclose(loop_data["temperature_C"].astype(float), temperature_C)
            & (loop_data["mode"].astype(str).str.upper() == mode)
        ]
        pristine = _extract_image_samples(group, "pristine")
        irradiated = _extract_image_samples(group, "irradiated")
        result = compare_image_samples_exact(pristine, irradiated)
        rows.append(
            {
                "temperature_C": float(temperature_C),
                "mode": str(mode),
                "n_images_pristine": len(pristine),
                "n_images_irradiated": len(irradiated),
                "n_loops_pristine": sum(sample.loop_count for sample in pristine),
                "n_loops_irradiated": sum(sample.loop_count for sample in irradiated),
                **result,
            }
        )

    results = pd.DataFrame(rows)
    for test_name in ("density", "shape", "joint"):
        results[f"{test_name}_q_bh"] = benjamini_hochberg(
            results[f"{test_name}_p_exact"].to_numpy(float)
        )
    results["resolved_at_alpha"] = results["joint_q_bh"] < float(alpha)
    results["interpretation"] = np.where(
        results["resolved_at_alpha"],
        "observable distributions differ",
        "difference not resolved with available image replication",
    )
    return results


def _fd_bin_edges(values_nm: np.ndarray, focus_quantile: float) -> np.ndarray:
    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm) & (values_nm > 0.0)]
    q25, q75 = np.quantile(values_nm, [0.25, 0.75])
    width = 2.0 * float(q75 - q25) * values_nm.size ** (-1.0 / 3.0)
    if not np.isfinite(width) or width <= 0.0:
        width = max(float(np.ptp(values_nm)) / max(10.0, np.sqrt(values_nm.size)), 0.05)
    x_max = max(width, np.ceil(np.quantile(values_nm, focus_quantile) / width) * width)
    return np.arange(0.0, x_max + 0.5 * width, width)


def _image_balanced_spectrum(samples: list[ImageSample], edges_nm: np.ndarray) -> np.ndarray:
    width_nm = np.diff(edges_nm)
    spectra = []
    for sample in samples:
        counts, _ = np.histogram(sample.diameters_nm, bins=edges_nm)
        spectra.append(counts / sample.volume_nm3 / width_nm)
    return np.mean(np.vstack(spectra), axis=0)


def plot_matched_density_spectra(
    loop_data: pd.DataFrame,
    results: pd.DataFrame,
    focus_quantile: float = 0.99,
):
    """Plot matched volume-normalized spectra on shared within-panel bins."""

    if not 0.0 < float(focus_quantile) <= 1.0:
        raise ValueError("focus_quantile must be in the interval (0, 1].")
    temperatures = sorted(results["temperature_C"].unique())
    modes = [mode for mode in ("DF", "BF") if mode in set(results["mode"])]
    figure, axes = plt.subplots(
        len(temperatures),
        len(modes),
        figsize=(6.1 * len(modes), 4.0 * len(temperatures)),
        squeeze=False,
    )
    colors = {"pristine": "#6a737d", "irradiated": "#ba0c2f"}
    for row, temperature_C in enumerate(temperatures):
        for column, mode in enumerate(modes):
            axis = axes[row, column]
            group = loop_data[
                np.isclose(loop_data["temperature_C"].astype(float), temperature_C)
                & (loop_data["mode"].astype(str).str.upper() == mode)
                & loop_data["series_id"].isin(["pristine", "irradiated"])
            ]
            if group.empty:
                axis.set_axis_off()
                continue
            samples = {
                series_id: _extract_image_samples(group, series_id)
                for series_id in ("pristine", "irradiated")
            }
            edges_nm = _fd_bin_edges(group["size"].to_numpy(float), focus_quantile)
            for series_id in ("pristine", "irradiated"):
                spectrum = _image_balanced_spectrum(samples[series_id], edges_nm)
                axis.stairs(
                    spectrum,
                    edges_nm,
                    linewidth=2.2,
                    color=colors[series_id],
                    label=series_id.capitalize(),
                )
            comparison = results[
                np.isclose(results["temperature_C"], temperature_C)
                & (results["mode"] == mode)
            ].iloc[0]
            axis.text(
                0.98,
                0.96,
                (
                    f"Density ratio={comparison['density_ratio']:.2f}\n"
                    f"W₁={comparison['wasserstein_nm']:.2f} nm; "
                    f"p_joint={comparison['joint_p_exact']:.3f}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize="small",
            )
            stage = "RT" if np.isclose(temperature_C, 25.0) else f"{temperature_C:g} °C"
            axis.set_title(f"{stage} — {mode}")
            axis.set_xlim(edges_nm[0], edges_nm[-1])
            axis.set_xlabel("Loop diameter (nm)")
            axis.set_ylabel(r"Loop density per diameter (nm$^{-4}$)")
            axis.grid(alpha=0.16)
            axis.legend(frameon=False, loc="upper left")
    figure.suptitle("Step 0: pristine versus irradiated observable loop distributions", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Each TEM image has equal weight; spectra are normalized by image volume and bin width.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    return figure


def plot_image_density_comparisons(loop_data: pd.DataFrame, results: pd.DataFrame):
    """Show the independent image-level number densities behind the tests."""

    temperatures = sorted(results["temperature_C"].unique())
    modes = [mode for mode in ("DF", "BF") if mode in set(results["mode"])]
    figure, axes = plt.subplots(
        len(temperatures),
        len(modes),
        figsize=(5.4 * len(modes), 3.8 * len(temperatures)),
        squeeze=False,
    )
    colors = {"pristine": "#6a737d", "irradiated": "#ba0c2f"}
    rng = np.random.default_rng(7)
    for row, temperature_C in enumerate(temperatures):
        for column, mode in enumerate(modes):
            axis = axes[row, column]
            group = loop_data[
                np.isclose(loop_data["temperature_C"].astype(float), temperature_C)
                & (loop_data["mode"].astype(str).str.upper() == mode)
                & loop_data["series_id"].isin(["pristine", "irradiated"])
            ]
            for x_position, series_id in enumerate(("pristine", "irradiated")):
                samples = _extract_image_samples(group, series_id)
                densities = np.asarray([sample.density_nm3 for sample in samples])
                jitter = rng.uniform(-0.055, 0.055, size=densities.size)
                axis.scatter(
                    np.full(densities.size, x_position) + jitter,
                    densities,
                    s=52,
                    color=colors[series_id],
                    zorder=3,
                )
                mean_density = float(np.mean(densities))
                axis.hlines(
                    mean_density,
                    x_position - 0.16,
                    x_position + 0.16,
                    color=colors[series_id],
                    linewidth=2.2,
                )
            comparison = results[
                np.isclose(results["temperature_C"], temperature_C)
                & (results["mode"] == mode)
            ].iloc[0]
            stage = "RT" if np.isclose(temperature_C, 25.0) else f"{temperature_C:g} °C"
            axis.set_title(
                f"{stage} — {mode}  (ratio={comparison['density_ratio']:.2f})"
            )
            axis.set_xticks([0, 1], ["Pristine", "Irradiated"])
            axis.set_yscale("log")
            axis.set_ylabel(r"Observed loop density (nm$^{-3}$)")
            axis.grid(axis="y", alpha=0.18)
    figure.suptitle("Independent image-level loop densities", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure
