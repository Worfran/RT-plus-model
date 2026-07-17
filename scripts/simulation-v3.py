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

    python scripts/simulation-v2.py --debug-only
    python scripts/simulation-v2.py --n-starts 1 --no-plot
    python scripts/simulation-v2.py --n-starts 8 --parallel-starts --max-workers 4 --no-plot
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make scripts/rtplus importable without installing a package.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rtplus.config import DataConfig, EVENT_SERIES, FitConfig, MaterialConstants
from rtplus.data_loader import load_all_loop_data,  print_dataset_summary
from rtplus.initial_conditions import describe_y0, make_series_initial_state
from rtplus.optimization import run_multistart
from rtplus.parameters import build_theta0_and_bounds, get_parameter_temperatures, unpack_theta
from rtplus.plotting import plot_event_series_results
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
    p.add_argument("--seed", type=int, default=10, help="Random seed for initial condition and randomized starts.")
    p.add_argument("--ic-strategy", choices=["random", "manual", "from_nonirradiated"], default="random", help="Initial-condition strategy.")
    p.add_argument("--t-end", type=float, default=3600.0, help="Simulation end time in seconds.")
    p.add_argument("--maxiter", type=int, default=2000, help="L-BFGS-B maximum iterations per start.")
    p.add_argument("--debug-only", action="store_true", help="Run theta0 forward simulation and stop before fitting.")
    p.add_argument("--no-plot", action="store_true", help="Skip plots after fitting.")
    return p.parse_args()


def main():
    args = parse_args()

    material = MaterialConstants()
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

    parameter_temperatures = get_parameter_temperatures(FIT_EVENT_SERIES)
    if args.temperatures is not None:
        requested = {float(T) for T in args.temperatures}
        parameter_temperatures = [T for T in parameter_temperatures if T in requested]
    print("\nTemperatures with fitted event parameters:", parameter_temperatures)

    initial_states = {
        series_id: make_series_initial_state(series_id, material, seed=args.seed)
        for series_id in FIT_EVENT_SERIES
    }
    for series_id, state in initial_states.items():
        print(f"\nSeries: {series_id}\n" + describe_y0(state, material))

    theta0, _ = build_theta0_and_bounds(parameter_temperatures)
    theta_debug = unpack_theta(theta0, parameter_temperatures)

    try:
        debug_predictions = simulate_all_series(
            event_series=FIT_EVENT_SERIES,
            theta=theta_debug,
            material=material,
            initial_states=initial_states,
        )
        print("\nForward simulation at theta0:")
        for series_id, events in debug_predictions.items():
            print(f"\nSeries: {series_id}")
            for event_order, prediction in events.items():
                print(
                    f"  Event {event_order}: T={prediction['temperature_C']:g} C, "
                    f"Df={2e7 * prediction['Rf']:.6f} nm, "
                    f"Dp={2e7 * prediction['Rp']:.6f} nm"
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
        FIT_EVENT_SERIES,
        initial_states,
        parameter_temperatures,
        fit_config,
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

    if best.objective >= fit_config.objective_fail_value:
        print("\nWARNING: best objective is the failure value. Run with --debug-only first.")
        return

    if not args.no_plot:
        plot_event_series_results(
            loop_data=loop_data,
            theta=best.theta,
            event_series=FIT_EVENT_SERIES,
            material=material,
            initial_states=initial_states,
        )


if __name__ == "__main__":
    main()
