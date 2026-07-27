"""Console reporting for fitted RT+ parameters and derived event values."""
from __future__ import annotations

from .config import ObservationConfig
from .observables import effective_bf_faulted_visibility, predicted_mean_diameters_nm


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
        ("D0", "Interstitial diffusion prefactor", _format_value(material.D0), "cm^2/s", "fixed"),
        ("Em", "Interstitial migration energy", _format_value(theta["Em"]), "eV", "fitted"),
        ("P0", "Perfect-loop coalescence prefactor", _format_value(theta["P0"]), "cm^3/s", "fitted"),
        ("Ea", "Perfect-loop coalescence barrier", _format_value(theta["Ea"]), "eV", "fitted"),
        ("P0_f", "Faulted-loop coalescence prefactor", _format_value(theta["P0_f"]), "cm^3/s", "fitted"),
        ("Ea_f", "Faulted-loop coalescence barrier", _format_value(theta["Ea_f"]), "eV", "fitted"),
        ("k_f", "Faulted distribution std/mean", _format_value(theta["k_f"]), "-", "fitted"),
        ("k_p", "Perfect distribution std/mean", _format_value(theta["k_p"]), "-", "fitted"),
        (
            "eta_BF,f",
            "BF faulted-loop detection efficiency",
            _format_value(theta.get("eta_bf_f", 1.0)),
            "-",
            "fitted",
        ),
    ]
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

    event_rows = []
    observation_config = ObservationConfig()
    visible_bf_faulted = effective_bf_faulted_visibility(
        theta,
        observation_config,
    )
    for event_order, prediction in sorted(predictions.items()):
        metadata = prediction.get("metadata", {})
        Df_nm, Dp_nm = predicted_mean_diameters_nm(prediction, theta)
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
            "Dp mean (nm)",
            "BF faulted fraction",
            "Di (cm^2/s)",
            "Pcs (cm^3/s)",
            "Pfcs (cm^3/s)",
            "Puf (1/s)",
        ),
        event_rows,
    )
