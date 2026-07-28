"""Modular replacement for simulation-v1.py.

This script is designed for your current project layout:

    RT-PLUS-MODEL/
    ├── Data/
    ├── notebooks/
    └── scripts/
        ├── BF-histograms.py
        ├── simulation-v1.py
        ├── simulation-v2.py
        └── rtplus/

Run from the project root:

    python scripts/simulation-v3.py --debug-only
    python scripts/simulation-v3.py --n-starts 1 --no-plot
    python scripts/simulation-v3.py --n-starts 8 --parallel-starts --max-workers 4 --no-plot
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make scripts/rtplus importable without installing a package.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtplus.config import DataConfig, EVENT_SERIES, FitConfig, MaterialConstants
from rtplus.data_loader import load_all_loop_data,  print_dataset_summary
from rtplus.diagnostics import print_bf_df_density_consistency
from rtplus.initial_conditions import describe_y0, fitted_initial_states
from rtplus.optimization import run_multistart
from rtplus.observables import (
    image_number_density_statistics,
    predicted_mean_diameters_nm,
    predicted_observed_number_density,
    theta_for_image_visibility,
)
from rtplus.parameters import build_theta0_and_bounds, get_parameter_temperatures, parameter_specs, unpack_theta
from rtplus.plotting import plot_meeting_results
from rtplus.reporting import print_final_parameter_tables
from rtplus.simulation import simulate_all_series
from rtplus.visibility_calibration import (
    calibrate_image_visibility,
    print_visibility_calibration,
)


# Fit only the irradiated specimen history for now.
FIT_EVENT_SERIES = {"irradiated": EVENT_SERIES["irradiated"]}


def parse_args():
    p = argparse.ArgumentParser(description="Run modular RT+ simulation/fitting workflow.")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Data", help="Directory containing BF/DF CSV files.")
    p.add_argument("--temperatures", nargs="*", type=float, default=None, help="Optional explicit fit temperatures, e.g. --temperatures 900 1100")
    p.add_argument("--exclude-rt", action="store_true", help="Exclude 25 °C / RT irradiated data from the fit.")
    p.add_argument("--n-starts", type=int, default=1, help="Number of optimizer initial guesses.")
    p.add_argument("--parallel-starts", action="store_true", help="Parallelize only optimizer starts. No temperature-level parallelism.")
    p.add_argument("--max-workers", type=int, default=None, help="Max workers for parallel starts.")
    p.add_argument("--seed", type=int, default=10, help="Random seed for randomized optimizer starts.")
    p.add_argument("--ic-strategy", choices=["fit"], default="fit", help="Fit a nonredundant initial state from RT and annealing data.")
    p.add_argument("--t-end", type=float, default=None, help="Optional duration override for every simulated event, in seconds.")
    p.add_argument("--enable-surface-sink", action="store_true", help="Enable the optional 100 nm two-surface point-defect sink extension.")
    p.add_argument("--enable-vacancy-extension", action="store_true", help="Fit Cv0 and Ev and enable Arrhenius vacancy recombination/loss.")
    p.add_argument("--maxiter", type=int, default=2000, help="L-BFGS-B maximum iterations per start.")
    p.add_argument(
        "--objective-mode",
        choices=["image_balanced_extended", "legacy"],
        default="image_balanced_extended",
        help=(
            "Fit per-image sizes and overdispersed image counts together "
            "(default), or use the previous pooled-size objective."
        ),
    )
    p.add_argument(
        "--count-overdispersion-floor",
        type=float,
        default=0.04,
        help="Minimum NB2 count overdispersion; 0.04 corresponds to 20%% image-level CV.",
    )
    p.add_argument(
        "--faulted-distribution",
        choices=["normal", "lognormal", "truncated_normal"],
        default="normal",
        help=(
            "Faulted-loop size family used consistently in DF and BF. "
            "Default: positive-centered Gaussian."
        ),
    )
    p.add_argument(
        "--no-smooth-visibility",
        action="store_true",
        help="Disable the smooth TEM small-loop detectability correction.",
    )
    p.add_argument("--df-rvis-nm", type=float, default=0.50, help="DF 50%% visibility radius in nm.")
    p.add_argument("--df-drvis-nm", type=float, default=0.15, help="DF visibility transition width in nm.")
    p.add_argument("--bf-rvis-nm", type=float, default=1.00, help="BF 50%% visibility radius in nm.")
    p.add_argument("--bf-drvis-nm", type=float, default=0.25, help="BF visibility transition width in nm.")
    p.add_argument(
        "--no-image-specific-visibility",
        action="store_true",
        help=(
            "Use only the mode-level visibility thresholds instead of "
            "pre-calibrated relative image thresholds."
        ),
    )
    p.add_argument(
        "--visibility-offset-sd-nm",
        type=float,
        default=0.20,
        help="Shrinkage scale for centered image visibility offsets in nm.",
    )
    p.add_argument(
        "--visibility-max-offset-nm",
        type=float,
        default=0.50,
        help="Maximum absolute image visibility offset in nm.",
    )
    p.add_argument("--debug-only", action="store_true", help="Run theta0 forward simulation and stop before fitting.")
    p.add_argument("--no-plot", action="store_true", help="Skip plots after fitting.")
    p.add_argument("--plot-dir", type=Path, default=PROJECT_ROOT / "Results", help="Directory for meeting-ready PNG plots.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.count_overdispersion_floor <= 0.0:
        raise ValueError("--count-overdispersion-floor must be positive.")
    if min(
        args.df_rvis_nm,
        args.df_drvis_nm,
        args.bf_rvis_nm,
        args.bf_drvis_nm,
        args.visibility_offset_sd_nm,
        args.visibility_max_offset_nm,
    ) <= 0.0:
        raise ValueError("Visibility radii, widths, and offset scales must be positive.")

    material = replace(
        MaterialConstants(),
        enable_surface_sink=args.enable_surface_sink,
        enable_vacancy_extension=args.enable_vacancy_extension,
    )
    fit_config = FitConfig(
        fit_temperatures=args.temperatures,
        exclude_room_temperature=args.exclude_rt,
        n_starts=args.n_starts,
        parallel_starts=args.parallel_starts,
        max_workers=args.max_workers,
        random_seed=args.seed,
        maxiter=args.maxiter,
        objective_mode=args.objective_mode,
        count_overdispersion_floor=args.count_overdispersion_floor,
        faulted_distribution=args.faulted_distribution,
        apply_smooth_visibility=not args.no_smooth_visibility,
        Rvis_DF_nm=args.df_rvis_nm,
        dRvis_DF_nm=args.df_drvis_nm,
        Rvis_BF_nm=args.bf_rvis_nm,
        dRvis_BF_nm=args.bf_drvis_nm,
        image_specific_visibility=not args.no_image_specific_visibility,
        visibility_offset_sd_nm=args.visibility_offset_sd_nm,
        visibility_max_offset_nm=args.visibility_max_offset_nm,
    )
    data_config = DataConfig(data_dir=args.data_dir)

    loop_data = load_all_loop_data(
        config=data_config,
        project_root=PROJECT_ROOT,
    )
    print_dataset_summary(loop_data)
    print_bf_df_density_consistency(loop_data)
    print(f"\nObjective mode: {fit_config.objective_mode}")

    event_series = FIT_EVENT_SERIES
    if args.t_end is not None:
        event_series = {
            series_id: [
                replace(event, duration_s=float(args.t_end)) if event.simulate else event
                for event in events
            ]
            for series_id, events in FIT_EVENT_SERIES.items()
        }

    if (
        fit_config.apply_smooth_visibility
        and fit_config.image_specific_visibility
    ):
        visibility_calibration = calibrate_image_visibility(
            loop_data,
            series_ids=event_series,
            base_rvis_by_mode_nm={
                "DF": fit_config.Rvis_DF_nm,
                "BF": fit_config.Rvis_BF_nm,
            },
            transition_by_mode_nm={
                "DF": fit_config.dRvis_DF_nm,
                "BF": fit_config.dRvis_BF_nm,
            },
            offset_sd_nm=fit_config.visibility_offset_sd_nm,
            max_offset_nm=fit_config.visibility_max_offset_nm,
        )
        fit_config = replace(
            fit_config,
            image_visibility_rvis_nm=visibility_calibration.rvis_by_image_nm,
            image_visibility_offsets_nm=visibility_calibration.offset_by_image_nm,
            image_visibility_efficiency=(
                visibility_calibration.efficiency_by_image
            ),
        )
        print_visibility_calibration(visibility_calibration)

    parameter_temperatures = get_parameter_temperatures(event_series)
    print("\nTemperatures with fitted event parameters:", parameter_temperatures)

    specs = parameter_specs(material.enable_vacancy_extension)
    theta0, _ = build_theta0_and_bounds(parameter_temperatures, specs=specs)
    theta_debug = unpack_theta(theta0, parameter_temperatures, specs=specs)
    theta_debug["faulted_distribution"] = fit_config.faulted_distribution
    if fit_config.apply_smooth_visibility:
        theta_debug.update(
            {
                "Rvis_DF_nm": fit_config.Rvis_DF_nm,
                "dRvis_DF_nm": fit_config.dRvis_DF_nm,
                "Rvis_BF_nm": fit_config.Rvis_BF_nm,
                "dRvis_BF_nm": fit_config.dRvis_BF_nm,
                "image_visibility_rvis_nm": dict(
                    fit_config.image_visibility_rvis_nm
                ),
                "image_visibility_offsets_nm": dict(
                    fit_config.image_visibility_offsets_nm
                ),
                "image_visibility_efficiency": dict(
                    fit_config.image_visibility_efficiency
                ),
            }
        )
    initial_states = fitted_initial_states(theta_debug, material, event_series)
    for series_id, state in initial_states.items():
        print(f"\nSeries: {series_id}\n" + describe_y0(state, material))

    try:
        debug_predictions = simulate_all_series(
            event_series=event_series,
            theta=theta_debug,
            material=material,
            initial_states=initial_states,
        )
        print("\nForward simulation at theta0:")
        for series_id, events in debug_predictions.items():
            print(f"\nSeries: {series_id}")
            for event_order, prediction in events.items():
                Df_nm, Dp_nm = predicted_mean_diameters_nm(
                    prediction,
                    theta_debug,
                )
                print(
                    f"  Event {event_order}: T={prediction['temperature_C']:g} C, "
                    f"Df_mean={Df_nm:.6f} nm, "
                    f"Dp_mean={Dp_nm:.6f} nm"
                )
    except Exception as exc:
        print("\nForward simulation failed before fitting.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        raise

    if args.debug_only:
        print("\nDebug-only mode: stopping before optimizer.")
        return

    best, all_results = run_multistart(
        loop_data,
        material,
        event_series,
        parameter_temperatures,
        fit_config,
        specs,
    )

    print("\nAll starts:")
    for r in all_results:
        print(f"  start={r.start_index}, success={r.success}, objective={r.objective:.6e}, message={r.message}")

    print("\nBest result:")
    print("  start:", best.start_index)
    print("  success:", best.success)
    print("  message:", best.message)
    print("  objective:", best.objective)
    print("  theta:", best.theta)

    best_initial_states = fitted_initial_states(best.theta, material, event_series)
    print("\nFitted initial state:")
    print(describe_y0(best_initial_states["irradiated"], material))

    best_predictions = simulate_all_series(
        event_series=event_series,
        theta=best.theta,
        material=material,
        initial_states=best_initial_states,
    )
    print_final_parameter_tables(
        theta=best.theta,
        material=material,
        predictions=best_predictions["irradiated"],
        objective=best.objective,
    )
    print("\nFitted event mean diameters:")
    for event_order, prediction in best_predictions["irradiated"].items():
        Df_nm, Dp_nm = predicted_mean_diameters_nm(prediction, best.theta)
        print(
            f"  event={event_order}, T={prediction['temperature_C']:g} C: "
            f"Df_mean={Df_nm:.3f} nm, Dp_mean={Dp_nm:.3f} nm"
        )
    print("\nPredicted versus measured observable number densities:")
    for event_order, prediction in best_predictions["irradiated"].items():
        for mode in ("DF", "BF"):
            group = loop_data[
                (loop_data["series_id"] == "irradiated")
                & (loop_data["event_order"] == event_order)
                & (loop_data["mode"] == mode)
            ]
            if group.empty:
                continue
            predicted_by_image = []
            for image_id, _ in group.groupby("image", sort=True):
                image_theta = theta_for_image_visibility(
                    best.theta,
                    series_id="irradiated",
                    event_order=event_order,
                    mode=mode,
                    image_id=image_id,
                )
                predicted_by_image.append(
                    predicted_observed_number_density(
                        mode,
                        prediction,
                        image_theta,
                    )
                )
            predicted_density = float(
                sum(predicted_by_image) / len(predicted_by_image)
            )
            observed_density, observed_std = image_number_density_statistics(
                group["image"].to_numpy(),
                group["volume_cm3"].to_numpy(dtype=float),
            )
            print(
                f"  event={event_order}, mode={mode}: "
                f"mean predicted={predicted_density:.3e} "
                f"[{min(predicted_by_image):.3e}, "
                f"{max(predicted_by_image):.3e}], "
                f"measured={observed_density:.3e} +/- {observed_std:.3e} cm^-3"
            )

    if best.objective >= fit_config.objective_fail_value:
        print("\nWARNING: best objective is the failure value. Run with --debug-only first.")
        return

    if not args.no_plot:
        plot_meeting_results(
            loop_data=loop_data,
            theta=best.theta,
            event_series=event_series,
            material=material,
            initial_states=best_initial_states,
            predictions=best_predictions["irradiated"],
            output_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
