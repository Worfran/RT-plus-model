"""Plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .observables import (
    lognormal_shape_from_mean_std,
    predicted_loop_pdf,
    predicted_mean_diameters_nm,
)
from .simulation import simulate_all_series, simulate_all_temperatures


def _prediction_value(prediction, name):
    return prediction[name] if isinstance(prediction, dict) else getattr(prediction, name)


def _lognormal_mode_from_mean_and_k(mean, k):
    """Mode of the lognormal whose arithmetic mean is ``mean`` and std is ``k*mean``."""
    sigma = lognormal_shape_from_mean_std(mean, k * mean)
    return float(mean * np.exp(-1.5 * sigma**2))


def plot_model_vs_data(values_nm, mode, prediction, theta, radius_unit_to_nm: float = 1e7, title: str = "", bins: int = 20):
    values_nm = np.asarray(values_nm, dtype=float)
    values_nm = values_nm[np.isfinite(values_nm)]
    values_nm = values_nm[values_nm > 0]
    if len(values_nm) == 0:
        print(f"No valid data for {title}")
        return

    Df_nm, Dp_nm = predicted_mean_diameters_nm(
        prediction,
        theta,
        radius_unit_to_nm,
    )

    x_max = max(float(values_nm.max()) * 1.2, Df_nm * 1.5, Dp_nm * 1.5, 1.0)
    x = np.linspace(1e-9, x_max, 500)
    pdf_model = predicted_loop_pdf(x, mode, prediction, theta, radius_unit_to_nm)

    plt.figure(figsize=(7, 5))
    plt.hist(values_nm, bins=bins, density=True, alpha=0.65, edgecolor="black", label="Experimental data")
    plt.plot(x, pdf_model, linewidth=2.5, label="RT+ fitted distribution")
    plt.axvline(Df_nm, linestyle="--", linewidth=1.5, label=f"Faulted diameter = {Df_nm:.2f} nm")
    if str(mode).upper() == "BF":
        plt.axvline(Dp_nm, linestyle=":", linewidth=1.5, label=f"Perfect diameter = {Dp_nm:.2f} nm")
    plt.title(title)
    plt.xlabel("Loop diameter (nm)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.tight_layout()
    


def plot_all(loop_data, theta, temperatures, material, sim_config, y0, bins: int = 20):
    predictions = simulate_all_temperatures(temperatures, theta, material, sim_config, y0)
    data_to_plot = loop_data[
        (loop_data["irradiated"] == True) &
        (loop_data["temperature_C"].isin([float(T) for T in temperatures]))
    ].copy()

    for (T_C, mode), group in data_to_plot.groupby(["temperature_C", "mode"]):
        plot_model_vs_data(
            values_nm=group["size"].to_numpy(dtype=float),
            mode=mode,
            prediction=predictions[float(T_C)],
            theta=theta,
            title=f"{float(T_C):g} °C - {mode} - irradiated",
            bins=bins,
        )
    plt.show()
    return predictions


def plot_event_series_results(
    loop_data,
    theta,
    event_series,
    material,
    initial_states,
    bins: int = 20,
    radius_unit_to_nm: float = 1e7,
):
    """Plot fitted BF/DF distributions for every event in both histories."""
    predictions = simulate_all_series(
        event_series=event_series,
        theta=theta,
        material=material,
        initial_states=initial_states,
    )
    figures = {}

    for series_id, events in event_series.items():
        ordered_events = sorted(events, key=lambda event: event.event_order)
        fig, axes = plt.subplots(
            len(ordered_events),
            2,
            figsize=(13, 4.2 * len(ordered_events)),
            squeeze=False,
            sharex=False,
        )

        for row, event in enumerate(ordered_events):
            prediction = predictions[series_id][event.event_order]
            Df_nm, Dp_nm = predicted_mean_diameters_nm(
                prediction,
                theta,
                radius_unit_to_nm,
            )

            for col, mode in enumerate(("DF", "BF")):
                ax = axes[row, col]
                group = loop_data[
                    (loop_data["series_id"] == series_id)
                    & (loop_data["event_order"] == event.event_order)
                    & (loop_data["mode"].str.upper() == mode)
                ]
                values_nm = group["size"].to_numpy(dtype=float)
                values_nm = values_nm[np.isfinite(values_nm) & (values_nm > 0)]

                if len(values_nm) == 0:
                    ax.text(0.5, 0.5, "No experimental data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                x_max = max(float(values_nm.max()) * 1.15, Df_nm * 1.5, Dp_nm * 1.5, 1.0)
                x = np.linspace(max(1e-6, x_max / 5000), x_max, 500)
                pdf_model = predicted_loop_pdf(x, mode, prediction, theta, radius_unit_to_nm)

                ax.hist(values_nm, bins=bins, density=True, alpha=0.6, edgecolor="black", label=f"Data (n={len(values_nm)})")
                ax.plot(x, pdf_model, linewidth=2.3, label="Fitted distribution")
                Df_mode_nm = _lognormal_mode_from_mean_and_k(Df_nm, theta["k_f"])
                ax.axvline(Df_nm, linestyle="--", linewidth=1.3, label=f"Faulted mean={Df_nm:.2f} nm")
                ax.axvline(Df_mode_nm, linestyle="-.", linewidth=1.1, label=f"Faulted mode={Df_mode_nm:.2f} nm")
                if mode == "BF":
                    Dp_mode_nm = _lognormal_mode_from_mean_and_k(Dp_nm, theta["k_p"])
                    ax.axvline(Dp_nm, linestyle=":", linewidth=1.5, label=f"Perfect mean={Dp_nm:.2f} nm")
                    ax.axvline(Dp_mode_nm, linestyle=(0, (3, 1, 1, 1)), linewidth=1.1, label=f"Perfect mode={Dp_mode_nm:.2f} nm")

                ax.set_title(f"Event {event.event_order}: {event.temperature_C:g} °C — {mode}")
                ax.set_xlabel("Loop diameter (nm)")
                ax.set_ylabel("Probability density")
                ax.legend(fontsize="small")

        fig.suptitle(f"RT+ fitted results — {series_id}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        figures[series_id] = fig

    plt.show()
    return predictions, figures
