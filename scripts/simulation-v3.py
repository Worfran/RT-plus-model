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
from rtplus.initial_conditions import describe_y0, fitted_initial_states
from rtplus.optimization import run_multistart
from rtplus.observables import (
    image_number_density_statistics,
    predicted_mean_diameters_nm,
    predicted_observed_number_density,
)
from rtplus.parameters import build_theta0_and_bounds, get_parameter_temperatures, parameter_specs, unpack_theta
from rtplus.plotting import plot_meeting_results
from rtplus.reporting import print_final_parameter_tables
from rtplus.simulation import simulate_all_series


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
    p.add_argument("--debug-only", action="store_true", help="Run theta0 forward simulation and stop before fitting.")
    p.add_argument("--no-plot", action="store_true", help="Skip plots after fitting.")
    p.add_argument("--plot-dir", type=Path, default=PROJECT_ROOT / "Results", help="Directory for meeting-ready PNG plots.")
    return p.parse_args()


def main():
    args = parse_args()

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
    )
    data_config = DataConfig(data_dir=args.data_dir)

    
    loop_data = load_all_loop_data(
        config=data_config,
        project_root=PROJECT_ROOT,
    )
    print_dataset_summary(loop_data)

    event_series = FIT_EVENT_SERIES
    if args.t_end is not None:
        event_series = {
            series_id: [
                replace(event, duration_s=float(args.t_end)) if event.simulate else event
                for event in events
            ]
            for series_id, events in FIT_EVENT_SERIES.items()
        }

    parameter_temperatures = get_parameter_temperatures(event_series)
    print("\nTemperatures with fitted event parameters:", parameter_temperatures)

    specs = parameter_specs(material.enable_vacancy_extension)
    theta0, _ = build_theta0_and_bounds(parameter_temperatures, specs=specs)
    theta_debug = unpack_theta(theta0, parameter_temperatures, specs=specs)
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
            predicted_density = predicted_observed_number_density(mode, prediction, best.theta)
            observed_density, observed_std = image_number_density_statistics(
                group["image"].to_numpy(),
                group["volume_cm3"].to_numpy(dtype=float),
            )
            print(
                f"  event={event_order}, mode={mode}: "
                f"predicted={predicted_density:.3e}, "
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
