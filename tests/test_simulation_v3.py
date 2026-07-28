from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rtplus.config import DataConfig, EVENT_SERIES, FitConfig, MaterialConstants, ObservationConfig
from rtplus.data_loader import load_all_loop_data
from rtplus.initial_conditions import fitted_initial_state, fitted_initial_states
from rtplus.objective import image_count_deviance, total_objective
from rtplus.observables import (
    binned_loop_number_density,
    binned_loop_number_density_from_images,
    image_number_density_statistics,
    predicted_loop_logpdf,
    predicted_loop_number_density_distribution,
    predicted_mean_radii_nm,
    predicted_observed_number_density,
)
from rtplus.ode import rhs
from rtplus.parameters import (
    build_theta0_and_bounds,
    faulted_width_at_temperature,
    get_parameter_temperatures,
    parameter_specs,
    unpack_theta,
)
from rtplus.physics import (
    compute_radius,
    lognormal_mean_radius_from_rms,
    lognormal_rms_radius_from_mean,
    loop_content_from_radius,
    loop_flux,
)
from rtplus.simulation import simulate_all_series
from rtplus.reporting import print_final_parameter_tables


class SimulationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = MaterialConstants()
        cls.event_series = {"irradiated": EVENT_SERIES["irradiated"]}
        cls.temperatures = get_parameter_temperatures(cls.event_series)
        cls.specs = parameter_specs(False)
        cls.theta_vec, _ = build_theta0_and_bounds(cls.temperatures, specs=cls.specs)
        cls.theta = unpack_theta(cls.theta_vec, cls.temperatures, specs=cls.specs)

    def test_fitted_initial_state_obeys_geometry(self):
        state = fitted_initial_state(self.theta, self.material)
        _, _, nf, np_, cf, cp = state
        rf = compute_radius(nf, cf, self.material.b, self.material.Omega0)
        rp = compute_radius(np_, cp, self.material.b, self.material.Omega0)
        self.assertAlmostEqual(
            lognormal_mean_radius_from_rms(
                rf,
                faulted_width_at_temperature(self.theta),
            )
            * 1e7,
            self.theta["Rf0_nm"],
            places=12,
        )
        self.assertAlmostEqual(
            lognormal_mean_radius_from_rms(rp, self.theta["k_p"]) * 1e7,
            self.theta["Rp0_nm"],
            places=12,
        )
        self.assertGreater(state[0], 1e10)

    def test_lognormal_mean_and_second_moment_are_consistent(self):
        mean_radius = 1.7e-7
        k = 0.65
        rms_radius = lognormal_rms_radius_from_mean(mean_radius, k)
        self.assertAlmostEqual(
            lognormal_mean_radius_from_rms(rms_radius, k),
            mean_radius,
        )
        prediction = {"Rf": rms_radius, "Rp": rms_radius, "Cf": 1.0, "Cp": 1.0}
        mean_f_nm, mean_p_nm = predicted_mean_radii_nm(
            prediction,
            {"k_f": k, "k_p": k},
        )
        self.assertAlmostEqual(mean_f_nm, mean_radius * 1e7)
        self.assertAlmostEqual(mean_p_nm, mean_radius * 1e7)

    def test_faulted_width_is_temperature_dependent(self):
        self.assertIn("k_f_initial", self.theta)
        self.assertEqual(
            set(self.theta["k_f_by_T"]),
            set(self.temperatures),
        )
        self.assertNotIn("k_f", self.theta)

        theta = {
            "k_f_initial": 0.2,
            "k_f_by_T": {900.0: 0.7, 1100.0: 0.4},
            "k_p": 0.5,
        }
        initial_prediction = {
            "temperature_C": 25.0,
            "metadata": {"simulated": False},
            "Rf": 1.0e-7,
            "Rp": 2.0e-7,
            "Cf": 8.0e16,
            "Cp": 4.0e16,
        }
        annealed_prediction = {
            **initial_prediction,
            "temperature_C": 900.0,
            "metadata": {"simulated": True},
        }

        self.assertAlmostEqual(
            faulted_width_at_temperature(theta),
            0.2,
        )
        self.assertAlmostEqual(
            faulted_width_at_temperature(theta, 900.0),
            0.7,
        )

        initial_mean_f, _ = predicted_mean_radii_nm(
            initial_prediction,
            theta,
        )
        annealed_mean_f, _ = predicted_mean_radii_nm(
            annealed_prediction,
            theta,
        )
        self.assertGreater(initial_mean_f, annealed_mean_f)

    def test_visibility_factors_enter_number_density(self):
        prediction = {"Rf": 1e-7, "Rp": 2e-7, "Cf": 8e16, "Cp": 4e16}
        theta = {"k_f": 0.5, "k_p": 0.5}
        cfg = ObservationConfig(
            relrod_resolution_radius_nm=0.0,
            bf_resolution_radius_nm=0.0,
        )
        df_density = predicted_observed_number_density("DF", prediction, theta, cfg)
        bf_density = predicted_observed_number_density("BF", prediction, theta, cfg)
        self.assertAlmostEqual(df_density, 0.25 * prediction["Cf"])
        self.assertAlmostEqual(bf_density, prediction["Cf"] + 0.5 * prediction["Cp"])

    def test_fitted_bf_faulted_detection_efficiency_changes_only_bf(self):
        prediction = {"Rf": 1e-7, "Rp": 2e-7, "Cf": 8e16, "Cp": 4e16}
        theta = {"k_f": 0.5, "k_p": 0.5, "eta_bf_f": 0.2}
        cfg = ObservationConfig(
            relrod_resolution_radius_nm=0.0,
            bf_resolution_radius_nm=0.0,
        )
        df_density = predicted_observed_number_density("DF", prediction, theta, cfg)
        bf_density = predicted_observed_number_density("BF", prediction, theta, cfg)
        self.assertAlmostEqual(df_density, 0.25 * prediction["Cf"])
        self.assertAlmostEqual(
            bf_density,
            0.2 * prediction["Cf"] + 0.5 * prediction["Cp"],
        )

    def test_binned_density_normalizes_by_bin_width_and_volume_density(self):
        values_nm = np.array([0.5, 1.5, 2.5])
        edges_nm = np.array([0.0, 1.0, 3.0])
        total_density = 9.0e16

        density = binned_loop_number_density(
            values_nm,
            total_density,
            edges_nm,
        )

        np.testing.assert_allclose(density, [3.0e16, 3.0e16])
        self.assertAlmostEqual(
            float(np.sum(density * np.diff(edges_nm))),
            total_density,
        )

    def test_image_resolved_density_uses_each_sampled_volume(self):
        values_nm = np.array([0.5, 1.5, 2.5])
        image_ids = np.array(["A", "A", "B"])
        volumes_cm3 = np.array([1.0e-15, 1.0e-15, 2.0e-15])
        edges_nm = np.array([0.0, 1.0, 3.0])

        mean_density, density_std = image_number_density_statistics(
            image_ids,
            volumes_cm3,
        )
        spectrum = binned_loop_number_density_from_images(
            values_nm,
            image_ids,
            volumes_cm3,
            edges_nm,
        )

        np.testing.assert_allclose(mean_density, 1.25e15, rtol=1e-14)
        np.testing.assert_allclose(density_std, 0.75e15, rtol=1e-14)
        np.testing.assert_allclose(
            float(np.sum(spectrum * np.diff(edges_nm))),
            mean_density,
            rtol=1e-14,
        )

    def test_model_density_spectrum_integrates_to_observable_density(self):
        prediction = {"Rf": 1.0e-7, "Rp": 2.0e-7, "Cf": 8.0e16, "Cp": 4.0e16}
        theta = {"k_f": 0.5, "k_p": 0.5}
        cfg = ObservationConfig(
            relrod_resolution_radius_nm=0.0,
            bf_resolution_radius_nm=0.0,
        )
        x_nm = np.geomspace(1.0e-4, 1.0e3, 20000)

        spectrum = predicted_loop_number_density_distribution(
            x_nm,
            "BF",
            prediction,
            theta,
            observation_config=cfg,
        )
        integrated_density = float(np.trapezoid(spectrum, x_nm))
        expected_density = predicted_observed_number_density(
            "BF",
            prediction,
            theta,
            cfg,
        )

        self.assertAlmostEqual(
            integrated_density / expected_density,
            1.0,
            places=6,
        )

    def test_resolution_cutoff_is_consistent_for_density_and_size_pdf(self):
        prediction = {"Rf": 1.0e-7, "Rp": 2.0e-7, "Cf": 8.0e16, "Cp": 4.0e16}
        theta = {"k_f": 0.5, "k_p": 0.5}
        cfg = ObservationConfig(
            bf_resolution_radius_nm=1.0,
            apply_resolution_cutoff=True,
        )
        x_nm = np.geomspace(1.0e-4, 1.0e3, 30000)
        spectrum = predicted_loop_number_density_distribution(
            x_nm,
            "BF",
            prediction,
            theta,
            observation_config=cfg,
        )
        expected_density = predicted_observed_number_density(
            "BF",
            prediction,
            theta,
            cfg,
        )
        np.testing.assert_allclose(
            float(np.trapezoid(spectrum, x_nm)) / expected_density,
            1.0,
            rtol=2.0e-4,
        )
        self.assertTrue(np.all(spectrum[x_nm < 2.0] == 0.0))

        conditional_pdf = np.exp(
            predicted_loop_logpdf(
                x_nm,
                "BF",
                prediction,
                theta,
                observation_config=cfg,
            )
        )
        np.testing.assert_allclose(
            float(np.trapezoid(conditional_pdf, x_nm)),
            1.0,
            rtol=2.0e-4,
        )

    def test_negative_binomial_count_deviance_uses_image_volumes(self):
        group = pd.DataFrame(
            {
                "image": ["A"] * 10 + ["B"] * 20,
                "volume_cm3": [1.0e-15] * 10 + [2.0e-15] * 20,
            }
        )
        matched_loss, alpha = image_count_deviance(
            group,
            predicted_density=1.0e16,
            overdispersion_floor=0.04,
        )
        mismatched_loss, _ = image_count_deviance(
            group,
            predicted_density=2.0e16,
            overdispersion_floor=0.04,
        )
        self.assertAlmostEqual(matched_loss, 0.0, places=12)
        self.assertAlmostEqual(alpha, 0.04)
        self.assertGreater(mismatched_loss, matched_loss)

    def test_events_are_sequential(self):
        states = fitted_initial_states(self.theta, self.material, self.event_series)
        predictions = simulate_all_series(self.event_series, self.theta, self.material, states)
        events = predictions["irradiated"]
        np.testing.assert_allclose(events[1]["y_initial"], events[0]["y_final"])
        np.testing.assert_allclose(events[2]["y_initial"], events[1]["y_final"])

    def test_loop_flux_matches_bawane_equation_s6(self):
        radius = 1.2e-7
        diffusivity = 3.4e-14
        concentration = 2.1e17
        expected = (
            2.0
            * np.pi**2
            * radius
            * diffusivity
            * concentration
            / np.log(8.0 * radius / self.material.r0)
        )
        self.assertAlmostEqual(
            loop_flux(radius, diffusivity, concentration, self.material.r0),
            expected,
        )

    def test_base_ode_conserves_mobile_and_stored_interstitials(self):
        rf = 1.0e-7
        rp = 2.0e-7
        cf = 8.0e16
        cp = 4.0e16
        state = np.array([
            1.4e17,
            0.0,
            loop_content_from_radius(rf, cf, self.material.b, self.material.Omega0),
            loop_content_from_radius(rp, cp, self.material.b, self.material.Omega0),
            cf,
            cp,
        ])
        params = {
            "a": self.material.a,
            "b": self.material.b,
            "Omega0": self.material.Omega0,
            "r0": self.material.r0,
            "Rii": self.material.Rii,
            "Ziv_iK": self.material.Ziv_iK,
            "Ziv_vK": self.material.Ziv_vK,
            "G0i": 0.0,
            "G0v": 0.0,
            "Di": 1.0e-14,
            "Dv": 0.0,
            "Puf": 2.0e-5,
            "Pcs": 3.0e-18,
            "Pfcs": 4.0e-18,
            "enable_vacancy_extension": False,
            "enable_surface_sink": False,
        }
        derivative = rhs(0.0, state, params)
        transfer_balance = derivative[0] + derivative[2] + derivative[3]
        scale = max(abs(derivative[0]), abs(derivative[2]), abs(derivative[3]))
        self.assertLessEqual(abs(transfer_balance), 1e-12 * scale)

    def test_faulted_coalescence_increases_radius_at_fixed_content(self):
        rf = 1.0e-7
        cf = 8.0e16
        state = np.array([
            0.0,
            0.0,
            loop_content_from_radius(rf, cf, self.material.b, self.material.Omega0),
            loop_content_from_radius(2.0e-7, 4.0e16, self.material.b, self.material.Omega0),
            cf,
            4.0e16,
        ])
        params = {
            "a": self.material.a,
            "b": self.material.b,
            "Omega0": self.material.Omega0,
            "r0": self.material.r0,
            "Rii": self.material.Rii,
            "Ziv_iK": self.material.Ziv_iK,
            "Ziv_vK": self.material.Ziv_vK,
            "G0i": 0.0,
            "G0v": 0.0,
            "Di": 0.0,
            "Dv": 0.0,
            "Puf": 0.0,
            "Pcs": 0.0,
            "Pfcs": 1.0e-18,
            "enable_vacancy_extension": False,
            "enable_surface_sink": False,
        }
        derivative = rhs(0.0, state, params)
        self.assertEqual(derivative[2], 0.0)
        self.assertLess(derivative[4], 0.0)
        dt = 0.1
        evolved = state + dt * derivative
        evolved_rf = compute_radius(
            evolved[2], evolved[4], self.material.b, self.material.Omega0
        )
        self.assertGreater(evolved_rf, rf)

    def test_corrected_rt_df_dataset_is_loaded(self):
        data = load_all_loop_data(DataConfig(data_dir=ROOT / "Data"), ROOT)
        selected = data[
            (data["series_id"] == "irradiated")
            & (data["event_order"] == 0)
            & (data["mode"] == "DF")
        ]
        self.assertEqual(len(selected), 148)
        self.assertAlmostEqual(float(selected["size"].mean()), 1.228162162162162, places=12)

    def test_duplicate_image_id_is_resolved_by_source_file(self):
        data = load_all_loop_data(DataConfig(data_dir=ROOT / "Data"), ROOT)
        irradiated = data[
            (data["source_file"] == "1100-DF-irr.csv")
            & (data["image"] == "Image 1135")
        ]
        pristine = data[
            (data["source_file"] == "RT-DF.csv")
            & (data["image"] == "Image 1135")
        ]

        self.assertTrue(len(irradiated) > 0)
        self.assertTrue(len(pristine) > 0)
        self.assertEqual(float(irradiated["volume_nm3_reference"].iloc[0]), 2767680.0)
        self.assertEqual(float(pristine["volume_nm3_reference"].iloc[0]), 2857680.0)
        self.assertAlmostEqual(
            float(irradiated["volume_cm3"].iloc[0]),
            2767680.0 * 1.25e-21,
        )

    def test_initial_objective_is_finite(self):
        data = load_all_loop_data(DataConfig(data_dir=ROOT / "Data"), ROOT)
        value = total_objective(
            self.theta_vec,
            data,
            self.material,
            self.event_series,
            self.temperatures,
            FitConfig(maxiter=1),
            self.specs,
        )
        self.assertTrue(np.isfinite(value))
        self.assertLess(value, 1e100)

    def test_final_parameter_report_contains_distribution_widths(self):
        from contextlib import redirect_stdout
        from io import StringIO

        states = fitted_initial_states(self.theta, self.material, self.event_series)
        predictions = simulate_all_series(self.event_series, self.theta, self.material, states)
        output = StringIO()
        with redirect_stdout(output):
            print_final_parameter_tables(
                self.theta,
                self.material,
                predictions["irradiated"],
                objective=1.23,
            )
        report = output.getvalue()
        self.assertIn("FINAL MODEL PARAMETERS", report)
        self.assertIn("k_f", report)
        self.assertIn("k_p", report)
        self.assertIn("eta_BF,f", report)
        self.assertIn("DERIVED EVENT VALUES", report)


if __name__ == "__main__":
    unittest.main()
