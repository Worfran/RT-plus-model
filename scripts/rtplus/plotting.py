"""Plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .observables import (
    binned_loop_number_density_from_images,
    predicted_loop_number_density_distribution,
    predicted_mean_diameters_nm,
)
from .simulation import simulate_all_series, simulate_all_temperatures
from .physics import diffusion_coeff

CM3_TO_NM3 = 1.0e-21


def _prediction_value(prediction, name):
    return prediction[name] if isinstance(prediction, dict) else getattr(prediction, name)


def _number_density_axis_label() -> str:
    return r"Loop density per diameter (nm$^{-4}$)"


def plot_model_vs_data(
    values_nm,
    mode,
    prediction,
    theta,
    image_ids,
    volume_nm3,
    radius_unit_to_nm: float = 1e7,
    title: str = "",
    bins: int = 20,
):
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
    model_density_nm4 = CM3_TO_NM3 * predicted_loop_number_density_distribution(
        x,
        mode,
        prediction,
        theta,
        radius_unit_to_nm=radius_unit_to_nm,
    )
    bin_edges = np.linspace(0.0, x_max, bins + 1)
    data_density = binned_loop_number_density_from_images(
        values_nm,
        image_ids,
        volume_nm3,
        bin_edges,
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        bin_edges[:-1],
        data_density,
        width=np.diff(bin_edges),
        align="edge",
        alpha=0.65,
        edgecolor="black",
        label="Experimental data",
    )
    plt.plot(x, model_density_nm4, linewidth=2.5, label="RT+ fitted distribution")
    plt.axvline(Df_nm, linestyle="--", linewidth=1.5, label=f"Faulted diameter = {Df_nm:.2f} nm")
    if str(mode).upper() == "BF":
        plt.axvline(Dp_nm, linestyle=":", linewidth=1.5, label=f"Perfect diameter = {Dp_nm:.2f} nm")
    plt.title(title)
    plt.xlabel("Loop diameter (nm)")
    plt.ylabel(_number_density_axis_label())
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
            image_ids=group["image"].to_numpy(),
            volume_nm3=group["volume_nm3_effective"].to_numpy(dtype=float),
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
    predictions=None,
    show: bool = True,
):
    """Plot fitted BF/DF distributions using common axes across events."""
    if predictions is None:
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
            sharex="col",
        )

        x_limits = {}
        for mode in ("DF", "BF"):
            mode_values = loop_data[
                (loop_data["series_id"] == series_id)
                & (loop_data["mode"].str.upper() == mode)
            ]["size"].to_numpy(dtype=float)
            mode_values = mode_values[np.isfinite(mode_values) & (mode_values > 0)]
            data_limit = float(np.quantile(mode_values, 0.995)) * 1.15
            model_means = []
            for event in ordered_events:
                Df_nm, Dp_nm = predicted_mean_diameters_nm(
                    predictions[series_id][event.event_order], theta, radius_unit_to_nm
                )
                model_means.append(Df_nm)
                if mode == "BF":
                    model_means.append(Dp_nm)
            x_limits[mode] = max(data_limit, 1.15 * max(model_means), 1.0)

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

                x_max = x_limits[mode]
                x = np.linspace(max(1e-6, x_max / 5000), x_max, 500)
                model_density_nm4 = CM3_TO_NM3 * predicted_loop_number_density_distribution(
                    x,
                    mode,
                    prediction,
                    theta,
                    radius_unit_to_nm=radius_unit_to_nm,
                )

                bin_edges = np.linspace(0.0, x_max, bins + 1)
                data_density = binned_loop_number_density_from_images(
                    values_nm,
                    group["image"].to_numpy(),
                    group["volume_nm3_effective"].to_numpy(dtype=float),
                    bin_edges,
                )
                ax.bar(
                    bin_edges[:-1],
                    data_density,
                    width=np.diff(bin_edges),
                    align="edge",
                    alpha=0.6,
                    edgecolor="#35586f",
                    color="#9ecae1",
                    label=f"Data (n={len(values_nm)})",
                )
                ax.plot(x, model_density_nm4, linewidth=2.3, color="#ba0c2f", label="RT+ model")
                ax.axvline(Df_nm, linestyle="--", linewidth=1.3, label=f"Faulted mean={Df_nm:.2f} nm")
                if mode == "BF":
                    ax.axvline(Dp_nm, color="#7b2cbf", linestyle=":", linewidth=1.5, label=f"Perfect mean={Dp_nm:.2f} nm")

                stage = "As-irradiated" if not event.simulate else f"{event.temperature_C:g} °C / {event.duration_s / 3600:g} h"
                ax.set_title(f"{stage} — {mode}")
                ax.set_xlabel("Loop diameter (nm)")
                ax.set_ylabel(_number_density_axis_label())
                ax.set_xlim(0.0, x_max)
                ax.grid(alpha=0.16)
                ax.legend(fontsize="small")

        fig.suptitle("Sequential loop-distribution fit on common diameter scales", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        figures[series_id] = fig

    if show:
        plt.show()
    return predictions, figures


def plot_distribution_progression(
    loop_data,
    event_series,
    series_id="irradiated",
    bins: int = 20,
):
    """Overlay measured number-density distributions across thermal events."""
    events = sorted(event_series[series_id], key=lambda event: event.event_order)
    colors = ("#6a737d", "#c54b18", "#ba0c2f")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for ax, mode in zip(axes, ("DF", "BF")):
        all_values = loop_data[
            (loop_data["series_id"] == series_id)
            & (loop_data["mode"].str.upper() == mode)
        ]["size"].to_numpy(dtype=float)
        all_values = all_values[np.isfinite(all_values) & (all_values > 0)]
        x_max = max(1.0, 1.15 * float(np.quantile(all_values, 0.995)))
        bin_edges = np.linspace(0.0, x_max, bins + 1)

        for index, event in enumerate(events):
            group = loop_data[
                (loop_data["series_id"] == series_id)
                & (loop_data["event_order"] == event.event_order)
                & (loop_data["mode"].str.upper() == mode)
            ]
            values = group["size"].to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            if len(values) == 0:
                continue
            label = "As-irradiated (25 °C)" if not event.simulate else f"{event.temperature_C:g} °C / {event.duration_s / 3600:g} h"
            color = colors[index % len(colors)]
            density_by_diameter = binned_loop_number_density_from_images(
                group["size"].to_numpy(dtype=float),
                group["image"].to_numpy(),
                group["volume_nm3_effective"].to_numpy(dtype=float),
                bin_edges,
            )
            ax.stairs(
                density_by_diameter,
                bin_edges,
                color=color,
                linewidth=2.2,
                label=label,
            )
            ax.axvline(float(np.mean(values)), color=color, linewidth=1.1, linestyle="--", alpha=0.8)

        ax.set_xlim(0.0, x_max)
        ax.set_xlabel("Loop diameter (nm)")
        ax.set_ylabel(_number_density_axis_label())
        ax.set_title(f"{mode}: measured number-density progression")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)

    fig.suptitle("Irradiated CeO₂: loop number-density distributions during sequential annealing", fontsize=14, fontweight="bold")
    return fig


def plot_arrhenius_diffusion(theta, material, event_series, series_id="irradiated"):
    """Plot the fitted interstitial Arrhenius law and annealing-stage values."""
    temperatures_C = sorted(
        float(event.temperature_C)
        for event in event_series[series_id]
        if event.simulate and event.duration_s > 0
    )
    if not temperatures_C:
        raise ValueError("At least one simulated annealing temperature is required.")

    lower = min(temperatures_C) - 50.0
    upper = max(temperatures_C) + 50.0
    line_C = np.linspace(lower, upper, 240)
    line_K = line_C + 273.15
    line_D = np.array([diffusion_coeff(material.D0, theta["Em"], T_K) for T_K in line_K])
    event_C = np.asarray(temperatures_C)
    event_K = event_C + 273.15
    event_D = np.array([diffusion_coeff(material.D0, theta["Em"], T_K) for T_K in event_K])

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(1000.0 / line_K, np.log10(line_D), color="#ba0c2f", linewidth=2.5, label="Fitted Arrhenius relation")
    ax.scatter(1000.0 / event_K, np.log10(event_D), s=65, color="#1f77b4", zorder=3)
    for T_C, x_value, y_value, D_value in zip(event_C, 1000.0 / event_K, np.log10(event_D), event_D):
        ax.annotate(f"{T_C:g} °C\nD={D_value:.2e} cm²/s", (x_value, y_value), xytext=(7, 8), textcoords="offset points")
    ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    ax.set_ylabel(r"$\log_{10}(D_i\,[\mathrm{cm^2/s}])$")
    ax.set_title("Interstitial diffusion follows Arrhenius dependence")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.04,
        rf"$D_i=D_0\exp(-E_m/k_BT)$" + "\n" + rf"$D_0={material.D0:.0e}$ cm²/s,  $E_m={theta['Em']:.3f}$ eV",
        transform=ax.transAxes,
    )
    return fig


def plot_meeting_results(
    loop_data,
    theta,
    event_series,
    material,
    initial_states,
    predictions,
    output_dir,
    show=True,
):
    """Create, save, and optionally show the three meeting-ready figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    progression = plot_distribution_progression(loop_data, event_series)
    _, fit_figures = plot_event_series_results(
        loop_data=loop_data,
        theta=theta,
        event_series=event_series,
        material=material,
        initial_states=initial_states,
        predictions={"irradiated": predictions},
        show=False,
    )
    arrhenius = plot_arrhenius_diffusion(theta, material, event_series)

    paths = {
        "experimental_progression": output_dir / "01-experimental-distribution-progression.png",
        "model_fit_progression": output_dir / "02-model-fit-progression.png",
        "arrhenius_diffusion": output_dir / "03-arrhenius-interstitial-diffusion.png",
    }
    progression.savefig(paths["experimental_progression"], dpi=300, bbox_inches="tight")
    fit_figures["irradiated"].savefig(paths["model_fit_progression"], dpi=300, bbox_inches="tight")
    arrhenius.savefig(paths["arrhenius_diffusion"], dpi=300, bbox_inches="tight")
    print("\nSaved meeting plots:")
    for path in paths.values():
        print(f"  {path}")
    if show:
        plt.show()
    return paths
