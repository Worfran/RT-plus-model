"""Calibrate image-specific TEM visibility thresholds and transition widths.

For image ``i`` the observation model is

    w_i(R) = 1 / (1 + exp(-(R - Rvis_i) / dRvis_i)).

Both parameters are inferred from size and volume-normalized count contrasts
between images at the same event and imaging mode, then frozen before the RT+
physical fit.  No image efficiency or amplitude is used: detectability
approaches one for large loops.
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
    drvis_by_image_nm: dict[VisibilityKey, float]
    width_log_offset_by_image: dict[VisibilityKey, float]
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


def _unpack_group_parameters(parameter_vector, n_images):
    """Unpack image threshold offsets and transition-width effects."""

    n_images = int(n_images)
    threshold_offsets_nm = np.asarray(
        parameter_vector[:n_images],
        dtype=float,
    )
    width_log_offsets = np.asarray(
        parameter_vector[n_images : 2 * n_images],
        dtype=float,
    )
    return threshold_offsets_nm, width_log_offsets


def _group_conditional_nll(
    parameter_vector,
    diameters_nm,
    image_indices,
    log_exposures,
    n_images,
    base_rvis_nm,
    base_transition_nm,
    offset_sd_nm,
    max_offset_nm,
    width_log_sd,
    min_width_nm,
    max_width_nm,
):
    """Conditional image-label likelihood with the shared size law canceled.

    At a fixed event and mode,

        P(image=i | D) proportional to V_i w_i(D).

    The known sampled volume ``V_i`` is the only exposure term. Therefore a
    volume-normalized count deficit must be explained by the image visibility
    curve instead of disappearing into an unconstrained image intercept.
    """

    offsets_nm, width_log_offsets = _unpack_group_parameters(
        parameter_vector,
        n_images,
    )
    radii_nm = 0.5 * np.asarray(diameters_nm, dtype=float)
    thresholds_nm = float(base_rvis_nm) + offsets_nm
    widths_nm = float(base_transition_nm) * np.exp(width_log_offsets)
    z = (radii_nm[:, None] - thresholds_nm[None, :]) / widths_nm[None, :]
    log_visibility = -np.logaddexp(0.0, -z)
    logits = np.asarray(log_exposures, dtype=float)[None, :] + log_visibility
    selected = logits[np.arange(len(image_indices)), image_indices]
    nll = -float(np.sum(selected - logsumexp(logits, axis=1)))

    # Partial pooling is essential because Rvis and dRvis can trade off over a
    # limited diameter range. The priors preserve the paper-scale mode anchor
    # while allowing a real image-specific count deficit to move the curve.
    nll += 0.5 * float(np.sum((offsets_nm / offset_sd_nm) ** 2))
    nll += 0.5 * float(np.sum((width_log_offsets / width_log_sd) ** 2))

    threshold_excess = np.maximum(np.abs(offsets_nm) - max_offset_nm, 0.0)
    nll += 1.0e5 * float(np.sum(threshold_excess**2))
    negative_threshold = np.maximum(1.0e-6 - thresholds_nm, 0.0)
    nll += 1.0e7 * float(np.sum(negative_threshold**2))
    too_narrow = np.maximum(float(min_width_nm) - widths_nm, 0.0)
    too_wide = np.maximum(widths_nm - float(max_width_nm), 0.0)
    nll += 1.0e7 * float(np.sum(too_narrow**2 + too_wide**2))
    return nll


def calibrate_image_visibility(
    loop_data: pd.DataFrame,
    *,
    series_ids,
    base_rvis_by_mode_nm,
    transition_by_mode_nm,
    offset_sd_nm: float = 0.20,
    max_offset_nm: float = 0.50,
    width_log_sd: float = 0.35,
    min_width_nm: float = 0.03,
    max_width_nm: float = 0.75,
) -> VisibilityCalibration:
    """Estimate image-specific ``Rvis`` and ``dRvis`` before RT+ fitting.

    Calibration is independent inside each ``(series, event, mode)`` group.
    Images in a group share the underlying physical size distribution, so
    conditioning on measured diameter cancels that unknown distribution. The
    image volume supplies the known exposure, allowing relative loop-count
    differences to inform detectability without an efficiency parameter.
    Single-image groups retain the mode-level threshold and width because no
    within-condition contrast is available.
    """

    scalar_settings = {
        "offset_sd_nm": offset_sd_nm,
        "max_offset_nm": max_offset_nm,
        "width_log_sd": width_log_sd,
        "min_width_nm": min_width_nm,
        "max_width_nm": max_width_nm,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in scalar_settings.values()):
        raise ValueError("Visibility calibration scales and bounds must be positive and finite.")
    if min_width_nm >= max_width_nm:
        raise ValueError("min_width_nm must be smaller than max_width_nm.")

    selected = loop_data[
        loop_data["series_id"].astype(str).isin({str(item) for item in series_ids})
    ].copy()
    rvis_by_image_nm: dict[VisibilityKey, float] = {}
    offset_by_image_nm: dict[VisibilityKey, float] = {}
    drvis_by_image_nm: dict[VisibilityKey, float] = {}
    width_log_offset_by_image: dict[VisibilityKey, float] = {}
    diagnostics = []

    grouped = selected.groupby(["series_id", "event_order", "mode"], sort=True)
    for (series_id, event_order, mode), group in grouped:
        mode = str(mode).strip().upper()
        base_rvis_nm = float(base_rvis_by_mode_nm[mode])
        base_transition_nm = float(transition_by_mode_nm[mode])
        if base_rvis_nm <= 0.0 or base_transition_nm <= 0.0:
            raise ValueError("Base visibility radii and widths must be positive.")
        if not min_width_nm <= base_transition_nm <= max_width_nm:
            raise ValueError(
                f"Base {mode} transition width must be between the configured bounds."
            )

        image_ids = sorted(group["image"].astype(str).unique())
        n_images = len(image_ids)
        if n_images == 0:
            continue

        offsets_nm = np.zeros(n_images, dtype=float)
        width_log_offsets = np.zeros(n_images, dtype=float)
        initial_nll = np.nan
        final_nll = np.nan
        success = True
        message = "single image; mode-level visibility retained"

        if n_images > 1:
            image_to_index = {
                image_id: index for index, image_id in enumerate(image_ids)
            }
            diameters_nm = group["size"].to_numpy(dtype=float)
            image_indices = np.array(
                [image_to_index[str(value)] for value in group["image"]],
                dtype=int,
            )
            volumes_nm3 = np.array(
                [
                    float(
                        group.loc[
                            group["image"].astype(str) == image_id,
                            "volume_nm3_effective",
                        ].iloc[0]
                    )
                    for image_id in image_ids
                ],
                dtype=float,
            )
            if np.any(~np.isfinite(volumes_nm3)) or np.any(volumes_nm3 <= 0.0):
                raise ValueError("Every visibility-calibration image needs a positive volume.")
            log_exposures = np.log(volumes_nm3)
            log_exposures -= float(np.mean(log_exposures))
            theta0 = np.zeros(2 * n_images, dtype=float)
            args = (
                diameters_nm,
                image_indices,
                log_exposures,
                n_images,
                base_rvis_nm,
                base_transition_nm,
                offset_sd_nm,
                max_offset_nm,
                width_log_sd,
                min_width_nm,
                max_width_nm,
            )
            initial_nll = _group_conditional_nll(theta0, *args)
            result = minimize(
                _group_conditional_nll,
                theta0,
                args=args,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
            )
            offsets_nm, width_log_offsets = _unpack_group_parameters(
                result.x,
                n_images,
            )
            # These walls are normally inactive; clip only as numerical safety.
            offsets_nm = np.clip(offsets_nm, -max_offset_nm, max_offset_nm)
            widths_nm = np.clip(
                base_transition_nm * np.exp(width_log_offsets),
                min_width_nm,
                max_width_nm,
            )
            width_log_offsets = np.log(widths_nm / base_transition_nm)
            success = bool(result.success)
            message = str(result.message)
            final_nll = float(result.fun)

        thresholds_nm = base_rvis_nm + offsets_nm
        widths_nm = base_transition_nm * np.exp(width_log_offsets)
        for image_id, offset_nm, threshold_nm, width_log_offset, width_nm in zip(
            image_ids,
            offsets_nm,
            thresholds_nm,
            width_log_offsets,
            widths_nm,
        ):
            key = visibility_image_key(series_id, event_order, mode, image_id)
            offset_by_image_nm[key] = float(offset_nm)
            rvis_by_image_nm[key] = max(1.0e-6, float(threshold_nm))
            width_log_offset_by_image[key] = float(width_log_offset)
            drvis_by_image_nm[key] = float(width_nm)

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
        drvis_by_image_nm=drvis_by_image_nm,
        width_log_offset_by_image=width_log_offset_by_image,
        diagnostics=tuple(diagnostics),
    )


def print_visibility_calibration(calibration: VisibilityCalibration) -> None:
    """Print the frozen image-specific thresholds and transition widths."""

    print("\nIMAGE-SPECIFIC VISIBILITY CALIBRATION")
    print(
        "  Rvis and dRvis use same-event/same-mode size and "
        "volume-normalized count contrasts; "
        "there is no image efficiency parameter."
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
                f"Rvis offset={calibration.offset_by_image_nm[key]:+.4f} nm, "
                f"Rvis={calibration.rvis_by_image_nm[key]:.4f} nm, "
                f"dRvis={calibration.drvis_by_image_nm[key]:.4f} nm"
            )
