"""Diagnostic helpers for data, initial conditions, and forward simulations."""
from __future__ import annotations

from .config import ObservationConfig
from .parameters import build_theta0_and_bounds, unpack_theta
from .observables import image_number_density_statistics, predicted_mean_diameters_nm
from .simulation import simulate_all_temperatures


def print_data_summary(loop_data):
    print("\nLoaded loop-size datasets:")
    print(loop_data.groupby(["temperature_C", "mode", "irradiated"]).size().rename("n_loops"))
    print("\nTotal measured loops:", len(loop_data))


def print_bf_df_density_consistency(loop_data, series_id="irradiated"):
    """Check whether nonnegative Cf and Cp can reproduce BF and DF counts.

    With resolution cutoffs disabled, the observation equations are

        C_DF = v_DF Cf
        C_BF = v_BF,f Cf + v_BF,p Cp.

    Since Cp cannot be negative, the faulted contribution inferred from DF is
    a lower bound on BF when eta_BF,f is one.  Violations quantify how much
    additional BF faulted-loop detection efficiency is needed.
    """

    cfg = ObservationConfig()
    if cfg.apply_resolution_cutoff:
        return

    selected = loop_data[loop_data["series_id"] == series_id]
    print("\nBF/DF observable-density consistency check at eta_BF,f = 1:")
    for event_order in sorted(selected["event_order"].unique()):
        densities = {}
        for mode in ("DF", "BF"):
            group = selected[
                (selected["event_order"] == event_order)
                & (selected["mode"].str.upper() == mode)
            ]
            if group.empty:
                continue
            densities[mode], _ = image_number_density_statistics(
                group["image"].to_numpy(),
                group["volume_cm3"].to_numpy(dtype=float),
            )
        if set(densities) != {"DF", "BF"}:
            continue

        implied_faulted_density = (
            densities["DF"] / cfg.relrod_faulted_visibility
        )
        minimum_bf_density = (
            cfg.bf_faulted_visibility * implied_faulted_density
        )
        ratio = minimum_bf_density / densities["BF"]
        status = "compatible" if ratio <= 1.0 else "INCOMPATIBLE"
        maximum_eta_if_no_perfect = min(1.0, 1.0 / ratio)
        print(
            f"  event={int(event_order)}: minimum BF from DF = "
            f"{minimum_bf_density:.3e}, measured BF = {densities['BF']:.3e}, "
            f"ratio={ratio:.2f} ({status}), "
            f"eta_BF,f must be <={maximum_eta_if_no_perfect:.3f} if Cp=0"
        )


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
    print_prediction_diagnostics(predictions, theta)
    return predictions


def print_prediction_diagnostics(predictions, theta):
    for T_C, pred in predictions.items():
        Df_nm, Dp_nm = predicted_mean_diameters_nm(pred, theta)
        print(f"\nT = {T_C:g} °C")
        print(f"  Faulted mean diameter = {Df_nm:.6f} nm")
        print(f"  Perfect mean diameter = {Dp_nm:.6f} nm")
        print(f"  Cf = {pred.Cf:.3e} cm^-3")
        print(f"  Cp = {pred.Cp:.3e} cm^-3")
        print(f"  Di = {pred.Di:.3e} cm^2/s")
        print(f"  Puf = {pred.Puf:.3e}")
        print(f"  Pcs = {pred.Pcs:.3e}")
