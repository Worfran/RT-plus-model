"""Diagnostic helpers for data, initial conditions, and forward simulations."""
from __future__ import annotations

from .parameters import build_theta0_and_bounds, unpack_theta
from .simulation import simulate_all_temperatures


def print_data_summary(loop_data):
    print("\nLoaded loop-size datasets:")
    print(loop_data.groupby(["temperature_C", "mode", "irradiated"]).size().rename("n_loops"))
    print("\nTotal measured loops:", len(loop_data))


def select_fit_temperatures(loop_data, fit_config) -> list[float]:
    if fit_config.fit_temperatures is not None:
        return [float(T) for T in fit_config.fit_temperatures]

    temps = sorted(float(T) for T in loop_data.loc[loop_data["irradiated"], "temperature_C"].dropna().unique())
    if fit_config.exclude_room_temperature:
        temps = [T for T in temps if T >= 100.0]
    return temps


def run_prefit_debug(loop_data, temperatures, material, sim_config, fit_config, y0):
    theta0, _ = build_theta0_and_bounds(temperatures)
    theta = unpack_theta(theta0, temperatures)
    predictions = simulate_all_temperatures(temperatures, theta, material, sim_config, y0)
    print("\nForward simulation at theta0:")
    print_prediction_diagnostics(predictions)
    return predictions


def print_prediction_diagnostics(predictions):
    for T_C, pred in predictions.items():
        print(f"\nT = {T_C:g} °C")
        print(f"  Rf diameter = {2*pred.Rf*1e7:.6f} nm")
        print(f"  Rp diameter = {2*pred.Rp*1e7:.6f} nm")
        print(f"  Cf = {pred.Cf:.3e} cm^-3")
        print(f"  Cp = {pred.Cp:.3e} cm^-3")
        print(f"  Di = {pred.Di:.3e} cm^2/s")
        print(f"  Puf = {pred.Puf:.3e}")
        print(f"  Pcs = {pred.Pcs:.3e}")
