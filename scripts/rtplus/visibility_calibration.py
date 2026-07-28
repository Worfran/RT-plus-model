"""Calibrate relative TEM visibility thresholds between comparable images.

The calibration has two frozen image terms:

1. a relative threshold inferred from size-composition contrasts, and
2. a large-loop detection efficiency inferred from volume-normalized counts.

Separating them is necessary when an image has fewer loops per volume but still
contains many small loops. A threshold alone cannot reproduce that pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


VisibilityKey = tuple[str, int, str, str]


@dataclass(frozen=True)
class VisibilityCalibration:
    """Frozen image-specific visibility values and group diagnostics."""

    rvis_by_image_nm: dict[VisibilityKey, float]
    offset_by_image_nm: dict[VisibilityKey, float]
    efficiency_by_image: dict[VisibilityKey, float]
    diagnostics: tuple[dict, ...]


def visibility_image_key(
    series_id,
    event_order,
    mode,
    image_id,
) -> VisibilityKey:
    """Return the canonical key used by fitting, plotting, and reporting."""

    return (
        str(series_id),
        int(event_order),
        str(mode).strip().upper(),
        str(image_id),
    )


def _centered_values(free_values: np.ndarray) -> np.ndarray:
    """Return an n-vector whose sum is exactly zero from n-1 values."""

    free_values = np.asarray(free_values, dtype=float)
    return np.concatenate((free_values, [-float(np.sum(free_values))]))


def _group_conditional_nll(
    parameter_vector,
    diameters_nm,
    image_indices,
    n_images,
    base_rvis_nm,
    transition_nm,
    offset_sd_nm,
    max_offset_nm,
):
    """Conditional image-label likelihood with the shared size law canceled.

    At a fixed event and mode,

        P(image=i | D) proportional to exp(a_i) w_i(D).

    The nuisance intercepts ``a_i`` absorb exposure and total-density
    differences. Therefore threshold offsets describe low-diameter depletion
    rather than being forced to explain uniform population differences.
    """

    n_free = n_images - 1
    offsets_nm = _centered_values(parameter_vector[:n_free])
    intercepts = np.concatenate(
        (np.asarray(parameter_vector[n_free:], dtype=float), [0.0])
    )
    radii_nm = 0.5 * np.asarray(diameters_nm, dtype=float)
    thresholds_nm = float(base_rvis_nm) + offsets_nm
    z = (
        radii_nm[:, None] - thresholds_nm[None, :]
    ) / float(transition_nm)
    log_visibility = -np.logaddexp(0.0, -z)
    logits = intercepts[None, :] + log_visibility
    selected = logits[np.arange(len(image_indices)), image_indices]
    nll = -float(np.sum(selected - logsumexp(logits, axis=1)))

    # Gaussian partial pooling prevents poorly resolved groups from producing
    # large image effects.  A smooth wall additionally enforces physical
    # thresholds and the user-selected maximum relative correction.
    nll += 0.5 * float(np.sum((offsets_nm / offset_sd_nm) ** 2))
    excess = np.maximum(np.abs(offsets_nm) - max_offset_nm, 0.0)
    nll += 1.0e5 * float(np.sum(excess**2))
    negative_threshold = np.maximum(1.0e-6 - thresholds_nm, 0.0)
    nll += 1.0e7 * float(np.sum(negative_threshold**2))
    return nll


def calibrate_image_visibility(
    loop_data: pd.DataFrame,
    *,
    series_ids,
    base_rvis_by_mode_nm,
    transition_by_mode_nm,
    offset_sd_nm: float = 0.20,
    max_offset_nm: float = 0.50,
) -> VisibilityCalibration:
    """Estimate centered image-specific visibility offsets before RT+ fitting.

    The calibration is performed independently inside each
    ``(series, event, mode)`` group.  Images in a group share the underlying
    physical size distribution; conditioning on measured diameter cancels that
    unknown distribution. A second frozen efficiency term is then derived from
    each image's count per sampled volume.
    """

    if offset_sd_nm <= 0.0 or not np.isfinite(offset_sd_nm):
        raise ValueError("offset_sd_nm must be positive and finite.")
    if max_offset_nm <= 0.0 or not np.isfinite(max_offset_nm):
        raise ValueError("max_offset_nm must be positive and finite.")

    selected = loop_data[
        loop_data["series_id"].astype(str).isin({str(item) for item in series_ids})
    ].copy()
    rvis_by_image_nm: dict[VisibilityKey, float] = {}
    offset_by_image_nm: dict[VisibilityKey, float] = {}
    efficiency_by_image: dict[VisibilityKey, float] = {}
    diagnostics = []

    grouped = selected.groupby(
        ["series_id", "event_order", "mode"],
        sort=True,
    )
    for (series_id, event_order, mode), group in grouped:
        mode = str(mode).strip().upper()
        base_rvis_nm = float(base_rvis_by_mode_nm[mode])
        transition_nm = float(transition_by_mode_nm[mode])
        if base_rvis_nm <= 0.0 or transition_nm <= 0.0:
            raise ValueError("Base visibility radii and widths must be positive.")

        image_ids = sorted(group["image"].astype(str).unique())
        n_images = len(image_ids)
        if n_images == 0:
            continue

        offsets_nm = np.zeros(n_images, dtype=float)
        initial_nll = np.nan
        final_nll = np.nan
        success = True
        message = "single image; no relative calibration"

        if n_images > 1:
            image_to_index = {
                image_id: index for index, image_id in enumerate(image_ids)
            }
            diameters_nm = group["size"].to_numpy(dtype=float)
            image_indices = np.array(
                [image_to_index[str(value)] for value in group["image"]],
                dtype=int,
            )
            n_free = n_images - 1
            theta0 = np.zeros(2 * n_free, dtype=float)
            initial_nll = _group_conditional_nll(
                theta0,
                diameters_nm,
                image_indices,
                n_images,
                base_rvis_nm,
                transition_nm,
                offset_sd_nm,
                max_offset_nm,
            )
            result = minimize(
                _group_conditional_nll,
                theta0,
                args=(
                    diameters_nm,
                    image_indices,
                    n_images,
                    base_rvis_nm,
                    transition_nm,
                    offset_sd_nm,
                    max_offset_nm,
                ),
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
            )
            offsets_nm = _centered_values(result.x[:n_free])
            offsets_nm = np.clip(
                offsets_nm,
                -float(max_offset_nm),
                float(max_offset_nm),
            )
            # Clipping is normally inactive. Recenter once so the reported and
            # applied group offsets retain the identifiability constraint.
            offsets_nm -= float(np.mean(offsets_nm))
            success = bool(result.success)
            message = str(result.message)
            final_nll = float(result.fun)

        group_thresholds = base_rvis_nm + offsets_nm
        pooled_radii_nm = 0.5 * group["size"].to_numpy(dtype=float)
        visibility_fractions = []
        observed_densities = []
        for image_id, threshold_nm in zip(image_ids, group_thresholds):
            z = (pooled_radii_nm - threshold_nm) / transition_nm
            visibility_fractions.append(
                float(np.mean(1.0 / (1.0 + np.exp(-z))))
            )
            image_data = group[group["image"].astype(str) == image_id]
            image_volumes = (
                image_data["volume_nm3_effective"]
                .drop_duplicates()
                .to_numpy(dtype=float)
            )
            if (
                image_volumes.size != 1
                or not np.isfinite(image_volumes[0])
                or image_volumes[0] <= 0.0
            ):
                raise ValueError(
                    f"Image {image_id!r} must have one positive volume."
                )
            observed_densities.append(
                float(len(image_data)) / float(image_volumes[0])
            )
        relative_physical_scores = (
            np.asarray(observed_densities, dtype=float)
            / np.maximum(np.asarray(visibility_fractions, dtype=float), 1.0e-12)
        )
        efficiencies = relative_physical_scores / float(
            np.max(relative_physical_scores)
        )
        efficiencies = np.clip(efficiencies, 0.05, 1.0)

        for image_id, offset_nm, efficiency in zip(
            image_ids,
            offsets_nm,
            efficiencies,
        ):
            key = visibility_image_key(
                series_id,
                event_order,
                mode,
                image_id,
            )
            offset_by_image_nm[key] = float(offset_nm)
            rvis_by_image_nm[key] = max(
                1.0e-6,
                base_rvis_nm + float(offset_nm),
            )
            efficiency_by_image[key] = float(efficiency)

        diagnostics.append(
            {
                "series_id": str(series_id),
                "event_order": int(event_order),
                "mode": mode,
                "n_images": int(n_images),
                "n_loops": int(len(group)),
                "initial_nll": float(initial_nll),
                "final_nll": float(final_nll),
                "success": success,
                "message": message,
            }
        )

    return VisibilityCalibration(
        rvis_by_image_nm=rvis_by_image_nm,
        offset_by_image_nm=offset_by_image_nm,
        efficiency_by_image=efficiency_by_image,
        diagnostics=tuple(diagnostics),
    )


def print_visibility_calibration(calibration: VisibilityCalibration) -> None:
    """Print the frozen relative corrections and group-fit diagnostics."""

    print("\nIMAGE-SPECIFIC VISIBILITY CALIBRATION")
    print(
        "  Thresholds use same-event/same-mode size contrasts; efficiencies "
        "use sampled volumes and loop-count differences."
    )
    for diagnostic in calibration.diagnostics:
        improvement = (
            diagnostic["initial_nll"] - diagnostic["final_nll"]
            if np.isfinite(diagnostic["initial_nll"])
            and np.isfinite(diagnostic["final_nll"])
            else np.nan
        )
        print(
            f"  {diagnostic['series_id']} event={diagnostic['event_order']} "
            f"{diagnostic['mode']}: images={diagnostic['n_images']}, "
            f"loops={diagnostic['n_loops']}, "
            f"conditional-NLL improvement={improvement:.4f}"
        )
        group_keys = sorted(
            key
            for key in calibration.rvis_by_image_nm
            if key[:3]
            == (
                diagnostic["series_id"],
                diagnostic["event_order"],
                diagnostic["mode"],
            )
        )
        for key in group_keys:
            print(
                f"    {key[3]}: "
                f"offset={calibration.offset_by_image_nm[key]:+.4f} nm, "
                f"Rvis={calibration.rvis_by_image_nm[key]:.4f} nm, "
                f"eta={calibration.efficiency_by_image[key]:.4f}"
            )
