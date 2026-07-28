"""Console reporting for fitted RT+ parameters and derived event values."""
from __future__ import annotations

import numpy as np

from .config import ObservationConfig
from .observables import (
    effective_bf_faulted_visibility,
    faulted_distribution_family,
    faulted_width_for_prediction,
    predicted_mean_diameters_nm,
    predicted_visible_mean_diameters_nm,
    visible_fraction_of_distribution,
)
from .parameters import faulted_width_at_temperature


def _format_value(value: float) -> str:
    value = float(value)
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude >= 1.0e4 or magnitude < 1.0e-3):
        return f"{value:.6e}"
    return f"{value:.6f}"


def _print_table(headers, rows) -> None:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    line = "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_final_parameter_tables(theta, material, predictions, objective=None) -> None:
    """Print fitted, initial-condition, and derived annealing parameters."""
    if objective is not None:
        print(f"\nFinal objective: {float(objective):.8g}")

    fitted_rows = [
        (
            "faulted family",
            "Faulted-loop size distribution",
            faulted_distribution_family(theta),
            "-",
            "selected",
        ),
        ("D0", "Interstitial diffusion prefactor", _format_value(material.D0), "cm^2/s", "fixed"),
        ("Em", "Interstitial migration energy", _format_value(theta["Em"]), "eV", "fitted"),
        ("P0", "Perfect-loop coalescence prefactor", _format_value(theta["P0"]), "cm^3/s", "fitted"),
        ("Ea", "Perfect-loop coalescence barrier", _format_value(theta["Ea"]), "eV", "fitted"),
        ("P0_f", "Faulted-loop coalescence prefactor", _format_value(theta["P0_f"]), "cm^3/s", "fitted"),
        ("Ea_f", "Faulted-loop coalescence barrier", _format_value(theta["Ea_f"]), "eV", "fitted"),
        (
            "k_f(as-irradiated)",
            "Faulted Gaussian sigma/center",
            _format_value(faulted_width_at_temperature(theta)),
            "-",
            "fitted",
        ),
        ("k_p", "Perfect distribution std/mean", _format_value(theta["k_p"]), "-", "fitted"),
        (
            "eta_BF,f",
            "BF faulted-loop detection efficiency",
            _format_value(theta.get("eta_bf_f", 1.0)),
            "-",
            "fitted",
        ),
    ]
    if "Rvis_DF_nm" in theta:
        fitted_rows.extend(
            [
                (
                    "Rvis_DF",
                    "DF base 50% visibility radius",
                    _format_value(theta["Rvis_DF_nm"]),
                    "nm",
                    "fixed",
                ),
                (
                    "dRvis_DF",
                    "DF visibility transition width",
                    _format_value(theta["dRvis_DF_nm"]),
                    "nm",
                    "fixed",
                ),
                (
                    "Rvis_BF",
                    "BF base 50% visibility radius",
                    _format_value(theta["Rvis_BF_nm"]),
                    "nm",
                    "fixed",
                ),
                (
                    "dRvis_BF",
                    "BF visibility transition width",
                    _format_value(theta["dRvis_BF_nm"]),
                    "nm",
                    "fixed",
                ),
            ]
        )
    for temperature_C, value in sorted(theta.get("k_f_by_T", {}).items()):
        fitted_rows.append(
            (
                f"k_f({temperature_C:g} C)",
                "Faulted Gaussian sigma/center",
                _format_value(value),
                "-",
                "fitted",
            )
        )
    for temperature_C, value in sorted(theta["Puf_by_T"].items()):
        fitted_rows.append(
            (f"Puf({temperature_C:g} C)", "Faulted-to-perfect conversion rate", _format_value(value), "1/s", "fitted")
        )
    if "Ev" in theta:
        fitted_rows.append(("Ev", "Vacancy migration energy", _format_value(theta["Ev"]), "eV", "fitted"))

    print("\nFINAL MODEL PARAMETERS")
    _print_table(("Parameter", "Meaning", "Value", "Units", "Status"), fitted_rows)

    initial_rows = [
        ("Ci0", _format_value(theta["Ci0"]), "cm^-3"),
        ("Cf0", _format_value(theta["Cf0"]), "cm^-3"),
        ("Cp0", _format_value(theta["Cp0"]), "cm^-3"),
        ("Rf0 mean", _format_value(theta["Rf0_nm"]), "nm"),
        ("Rp0 mean", _format_value(theta["Rp0_nm"]), "nm"),
    ]
    if "Cv0" in theta:
        initial_rows.insert(1, ("Cv0", _format_value(theta["Cv0"]), "cm^-3"))
    print("\nFITTED INITIAL CONDITIONS")
    _print_table(("Parameter", "Value", "Units"), initial_rows)

    image_thresholds = theta.get("image_visibility_rvis_nm", {})
    image_offsets = theta.get("image_visibility_offsets_nm", {})
    image_efficiencies = theta.get("image_visibility_efficiency", {})
    if image_thresholds:
        calibration_rows = []
        for key, radius_nm in sorted(image_thresholds.items()):
            series_id, event_order, mode, image_id = key
            calibration_rows.append(
                (
                    series_id,
                    event_order,
                    mode,
                    image_id,
                    _format_value(image_offsets.get(key, 0.0)),
                    _format_value(radius_nm),
                    _format_value(image_efficiencies.get(key, 1.0)),
                )
            )
        print("\nFROZEN IMAGE-SPECIFIC VISIBILITY VALUES")
        _print_table(
            (
                "Series",
                "Event",
                "Mode",
                "Image",
                "Offset (nm)",
                "Rvis (nm)",
                "Efficiency",
            ),
            calibration_rows,
        )

    event_rows = []
    observation_config = ObservationConfig()
    visible_bf_faulted = effective_bf_faulted_visibility(
        theta,
        observation_config,
    )
    for event_order, prediction in sorted(predictions.items()):
        metadata = prediction.get("metadata", {})
        Df_nm, Dp_nm = predicted_mean_diameters_nm(prediction, theta)
        k_f = faulted_width_for_prediction(prediction, theta)
        visible_faulted_density = visible_bf_faulted * prediction["Cf"]
        visible_perfect_density = (
            observation_config.bf_perfect_visibility * prediction["Cp"]
        )
        bf_faulted_fraction = visible_faulted_density / (
            visible_faulted_density + visible_perfect_density
        )
        event_rows.append(
            (
                event_order,
                f"{prediction['temperature_C']:g}",
                _format_value(Df_nm),
                _format_value(k_f),
                _format_value(Dp_nm),
                _format_value(bf_faulted_fraction),
                _format_value(metadata.get("Di", 0.0)) if metadata.get("simulated") else "initial state",
                _format_value(metadata.get("Pcs", 0.0)) if metadata.get("simulated") else "-",
                _format_value(metadata.get("Pfcs", 0.0)) if metadata.get("simulated") else "-",
                _format_value(metadata.get("Puf", 0.0)) if metadata.get("simulated") else "-",
            )
        )
    print("\nDERIVED EVENT VALUES")
    _print_table(
        (
            "Event",
            "T (C)",
            "Df mean (nm)",
            "k_f",
            "Dp mean (nm)",
            "BF faulted fraction",
            "Di (cm^2/s)",
            "Pcs (cm^3/s)",
            "Pfcs (cm^3/s)",
            "Puf (1/s)",
        ),
        event_rows,
    )

    visibility_rows = []
    family_f = faulted_distribution_family(theta)
    image_thresholds = theta.get("image_visibility_rvis_nm", {})
    image_efficiencies = theta.get("image_visibility_efficiency", {})

    def local_thetas(event_order, mode):
        selected_items = [
            (key, radius_nm)
            for key, radius_nm in image_thresholds.items()
            if key[0] == "irradiated"
            and key[1] == event_order
            and key[2] == mode
        ]
        if not selected_items:
            return [theta]
        results = []
        for key, radius_nm in selected_items:
            local_theta = dict(theta)
            local_theta[f"Rvis_{mode}_nm"] = float(radius_nm)
            local_theta[f"visibility_efficiency_{mode}"] = float(
                image_efficiencies.get(key, 1.0)
            )
            results.append(local_theta)
        return results

    for event_order, prediction in sorted(predictions.items()):
        Df_nm, Dp_nm = predicted_mean_diameters_nm(prediction, theta)
        k_f = faulted_width_for_prediction(prediction, theta)
        df_thetas = local_thetas(event_order, "DF")
        bf_thetas = local_thetas(event_order, "BF")
        df_visible_means = [
            predicted_visible_mean_diameters_nm(
                "DF",
                prediction,
                local_theta,
            )
            for local_theta in df_thetas
        ]
        bf_visible_means = [
            predicted_visible_mean_diameters_nm(
                "BF",
                prediction,
                local_theta,
            )
            for local_theta in bf_thetas
        ]
        visible_Df_DF_nm = float(np.mean([item[0] for item in df_visible_means]))
        visible_Df_BF_nm = float(np.mean([item[0] for item in bf_visible_means]))
        visible_Dp_BF_nm = float(np.mean([item[1] for item in bf_visible_means]))
        fraction_Df_DF = float(
            np.mean(
                [
                    visible_fraction_of_distribution(
                        Df_nm,
                        k_f,
                        family_f,
                        "DF",
                        local_theta,
                    )
                    for local_theta in df_thetas
                ]
            )
        )
        fraction_Df_BF = float(
            np.mean(
                [
                    visible_fraction_of_distribution(
                        Df_nm,
                        k_f,
                        family_f,
                        "BF",
                        local_theta,
                    )
                    for local_theta in bf_thetas
                ]
            )
        )
        fraction_Dp_BF = float(
            np.mean(
                [
                    visible_fraction_of_distribution(
                        Dp_nm,
                        theta["k_p"],
                        "lognormal",
                        "BF",
                        local_theta,
                    )
                    for local_theta in bf_thetas
                ]
            )
        )
        visibility_rows.append(
            (
                event_order,
                f"{prediction['temperature_C']:g}",
                _format_value(fraction_Df_DF),
                _format_value(visible_Df_DF_nm),
                _format_value(fraction_Df_BF),
                _format_value(visible_Df_BF_nm),
                _format_value(fraction_Dp_BF),
                _format_value(visible_Dp_BF_nm),
            )
        )

    print("\nDERIVED IMAGE-AVERAGED VISIBILITY VALUES")
    _print_table(
        (
            "Event",
            "T (C)",
            "DF <w_f>",
            "DF visible Df (nm)",
            "BF <w_f>",
            "BF visible Df (nm)",
            "BF <w_p>",
            "BF visible Dp (nm)",
        ),
        visibility_rows,
    )
