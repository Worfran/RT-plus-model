from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rtplus.config import DataConfig, EVENT_SERIES, FitConfig, MaterialConstants, ObservationConfig
from rtplus.data_loader import load_all_loop_data
from rtplus.initial_conditions import fitted_initial_state, fitted_initial_states
from rtplus.objective import (
    faulted_size_fit_fraction_for_prediction,
    image_count_deviance,
    total_objective,
    upper_trimmed_size_subset,
)
from rtplus.observables import (
    binned_loop_number_density,
    binned_loop_number_density_from_images,
    faulted_distribution_family,
    image_number_density_statistics,
    loop_size_logpdf,
    positive_centered_normal_parameters,
    predicted_loop_logpdf,
    predicted_loop_number_density_distribution,
    predicted_mean_radii_nm,
    predicted_observed_number_density,
    theta_for_image_visibility,
    truncated_normal_parameters_from_mean_and_k,
    visible_fraction_of_distribution,
    visibility_log_weight,
)
from rtplus.ode import rhs
from rtplus.optimization import make_start_vectors
from rtplus.parameters import (
    build_theta0_and_bounds,
    faulted_width_at_temperature,
    get_parameter_temperatures,
    parameter_specs,
    unpack_theta,
)
from rtplus.physics import (
    COALESCENCE_LIFETIME_DENSITY_EXPONENT,
    COALESCENCE_RADIUS_EXPONENT,
    coalescence_inverse_lifetime,
    coalescence_number_loss,
    compute_radius,
    lognormal_mean_radius_from_rms,
    lognormal_rms_radius_from_mean,
    loop_content_from_radius,
    loop_flux,
)
from rtplus.simulation import simulate_all_series
from rtplus.reporting import print_final_parameter_tables
from rtplus.visibility_calibration import (
    calibrate_image_visibility,
)


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
            rf * 1e7,
            self.theta["Rf0_nm"],
            places=12,
        )
        self.assertAlmostEqual(
            rp * 1e7,
            self.theta["Rp0_nm"],
            places=12,
        )
        self.assertGreater(state[0], 1e10)

    def test_multistart_overshoots_existing_initial_populations_only(self):
        fit_config = FitConfig(
            n_starts=4,
            random_seed=3,
            initial_population_start_multipliers=(1.0, 3.0, 10.0),
            initial_population_min_multiplier=1.0,
        )
        starts, _ = make_start_vectors(
            self.temperatures,
            fit_config,
            self.specs,
        )
        baseline = unpack_theta(starts[0], self.temperatures, specs=self.specs)
        moderate = unpack_theta(starts[1], self.temperatures, specs=self.specs)
        strong = unpack_theta(starts[2], self.temperatures, specs=self.specs)

        self.assertEqual(len(starts), 4)
        self.assertEqual(len(starts[0]), len(self.theta_vec))
        for name in ("Ci0", "Cf0", "Cp0"):
            self.assertAlmostEqual(moderate[name] / baseline[name], 3.0)
            self.assertAlmostEqual(strong[name] / baseline[name], 10.0)
        for name in ("Rf0_nm", "Rp0_nm"):
            self.assertAlmostEqual(moderate[name], baseline[name])
            self.assertAlmostEqual(strong[name], baseline[name])

    def test_initial_population_minimum_multiplier_changes_fit_bounds(self):
        fit_config = FitConfig(
            n_starts=3,
            initial_population_start_multipliers=(10.0, 30.0, 100.0),
            initial_population_min_multiplier=10.0,
        )
        starts, bounds = make_start_vectors(
            self.temperatures,
            fit_config,
            self.specs,
        )
        for start in starts:
            theta = unpack_theta(start, self.temperatures, specs=self.specs)
            for name in ("Ci0", "Cf0", "Cp0"):
                self.assertGreaterEqual(
                    theta[name] / self.theta[name],
                    10.0 * (1.0 - 1.0e-12),
                )

        lower_bound_theta = unpack_theta(
            np.asarray([low for low, _ in bounds]),
            self.temperatures,
            specs=self.specs,
        )
        for name in ("Ci0", "Cf0", "Cp0"):
            self.assertAlmostEqual(
                lower_bound_theta[name] / self.theta[name],
                10.0,
            )

    def test_default_rt_weight_is_relaxed_but_still_emphasized(self):
        self.assertAlmostEqual(FitConfig().room_temperature_loss_weight, 1.5)

    def test_default_absolute_count_constraint_retains_both_modes(self):
        self.assertEqual(
            tuple(FitConfig().absolute_count_modes),
            ("BF", "DF"),
        )

    def test_legacy_coalescence_uses_quadratic_density_loss(self):
        coefficient = 2.5e-18
        density = 3.2e16
        loss = coalescence_number_loss(
            coefficient,
            1.4e-7,
            density,
            model="legacy_quadratic",
        )
        self.assertAlmostEqual(loss, coefficient * density**2)

    def test_legacy_coalescence_parameter_specs_restore_old_scale(self):
        legacy_specs = parameter_specs(
            False,
            coalescence_model="legacy_quadratic",
        )
        by_name = {spec.name: spec for spec in legacy_specs}
        self.assertEqual(by_name["P0"].initial, 1.0e-12)
        self.assertEqual(by_name["P0_f"].bounds, (1.0e-30, 1.0e-6))

    def test_upper_trimmed_size_subset_keeps_the_small_loop_tail(self):
        values = np.arange(1.0, 101.0)
        retained, lower, upper = upper_trimmed_size_subset(values, 0.95)
        self.assertEqual(lower, -np.inf)
        self.assertAlmostEqual(upper, 95.05)
        np.testing.assert_array_equal(retained, np.arange(1.0, 96.0))
        all_values, lower, upper = upper_trimmed_size_subset(values, 1.0)
        np.testing.assert_array_equal(all_values, values)
        self.assertEqual(lower, -np.inf)
        self.assertEqual(upper, np.inf)

    def test_1100_df_uses_the_complete_size_distribution(self):
        fit_config = FitConfig(
            faulted_size_fit_fraction=0.95,
            faulted_full_distribution_temperatures=(1100.0,),
        )
        self.assertAlmostEqual(
            faulted_size_fit_fraction_for_prediction(
                {"temperature_C": 900.0},
                fit_config,
            ),
            0.95,
        )
        self.assertAlmostEqual(
            faulted_size_fit_fraction_for_prediction(
                {"temperature_C": 1100.0},
                fit_config,
            ),
            1.0,
        )

    def test_observation_center_uses_source_representative_radius(self):
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
        self.assertAlmostEqual(mean_f_nm, rms_radius * 1e7)
        self.assertAlmostEqual(mean_p_nm, rms_radius * 1e7)

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
        self.assertAlmostEqual(initial_mean_f, annealed_mean_f)

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

    def test_smooth_visibility_suppresses_small_loops_consistently(self):
        prediction = {"Rf": 1e-7, "Rp": 2e-7, "Cf": 8e16, "Cp": 4e16}
        theta = {
            "k_f": 0.5,
            "k_p": 0.5,
            "Rvis_DF_nm": 1.0,
            "dRvis_DF_nm": 0.2,
        }
        weights = np.exp(
            visibility_log_weight(
                np.array([0.5, 2.0, 5.0]),
                "DF",
                theta,
            )
        )
        self.assertTrue(np.all(np.diff(weights) > 0.0))
        corrected_density = predicted_observed_number_density(
            "DF",
            prediction,
            theta,
        )
        self.assertGreater(corrected_density, 0.0)
        self.assertLess(corrected_density, 0.25 * prediction["Cf"])

        x_nm = np.geomspace(1.0e-4, 1.0e3, 30000)
        spectrum = predicted_loop_number_density_distribution(
            x_nm,
            "DF",
            prediction,
            theta,
        )
        np.testing.assert_allclose(
            float(np.trapezoid(spectrum, x_nm)) / corrected_density,
            1.0,
            rtol=2.0e-4,
        )

    def test_image_specific_visibility_activates_selected_threshold_and_width(self):
        key = ("irradiated", 1, "DF", "Image B")
        theta = {
            "Rvis_DF_nm": 0.5,
            "dRvis_DF_nm": 0.15,
            "image_visibility_rvis_nm": {key: 0.8},
            "image_visibility_drvis_nm": {key: 0.31},
        }
        image_a = theta_for_image_visibility(
            theta,
            series_id="irradiated",
            event_order=1,
            mode="DF",
            image_id="Image A",
        )
        image_b = theta_for_image_visibility(
            theta,
            series_id="irradiated",
            event_order=1,
            mode="DF",
            image_id="Image B",
        )
        self.assertIs(image_a, theta)
        self.assertAlmostEqual(image_b["Rvis_DF_nm"], 0.8)
        self.assertAlmostEqual(image_b["dRvis_DF_nm"], 0.31)
        self.assertAlmostEqual(theta["Rvis_DF_nm"], 0.5)
        self.assertAlmostEqual(theta["dRvis_DF_nm"], 0.15)

    def test_same_event_calibration_recovers_relative_visibility_order(self):
        rng = np.random.default_rng(4)
        physical_diameters = rng.lognormal(
            mean=np.log(2.5),
            sigma=0.45,
            size=30000,
        )
        rows = []
        for image_id, threshold_nm in (
            ("Image A", 0.30),
            ("Image B", 0.80),
        ):
            probability = 1.0 / (
                1.0
                + np.exp(
                    -(
                        0.5 * physical_diameters - threshold_nm
                    )
                    / 0.15
                )
            )
            accepted = physical_diameters[
                rng.random(len(physical_diameters)) < probability
            ][:4000]
            rows.extend(
                {
                    "series_id": "irradiated",
                    "event_order": 1,
                    "mode": "DF",
                    "image": image_id,
                    "size": diameter,
                    "volume_nm3_effective": 1.0e6,
                }
                for diameter in accepted
            )
        calibration = calibrate_image_visibility(
            pd.DataFrame(rows),
            series_ids={"irradiated"},
            base_rvis_by_mode_nm={"DF": 0.5},
            transition_by_mode_nm={"DF": 0.15},
            offset_sd_nm=0.2,
            max_offset_nm=0.5,
        )
        key_a = ("irradiated", 1, "DF", "Image A")
        key_b = ("irradiated", 1, "DF", "Image B")
        self.assertLess(
            calibration.rvis_by_image_nm[key_a],
            calibration.rvis_by_image_nm[key_b],
        )
        self.assertLessEqual(
            abs(calibration.offset_by_image_nm[key_a]),
            0.5,
        )
        self.assertLessEqual(
            abs(calibration.offset_by_image_nm[key_b]),
            0.5,
        )
        self.assertGreater(calibration.drvis_by_image_nm[key_a], 0.0)
        self.assertGreater(calibration.drvis_by_image_nm[key_b], 0.0)

    def test_visibility_calibration_uses_volume_normalized_count_deficit(self):
        common_sizes = np.linspace(1.0, 4.0, 500)
        rows = []
        for image_id, volume_nm3 in (
            ("Image A", 1.0e6),
            ("Image B", 2.0e6),
        ):
            rows.extend(
                {
                    "series_id": "irradiated",
                    "event_order": 1,
                    "mode": "DF",
                    "image": image_id,
                    "size": diameter,
                    "volume_nm3_effective": volume_nm3,
                }
                for diameter in common_sizes
            )
        calibration = calibrate_image_visibility(
            pd.DataFrame(rows),
            series_ids={"irradiated"},
            base_rvis_by_mode_nm={"DF": 0.5},
            transition_by_mode_nm={"DF": 0.15},
            offset_sd_nm=0.2,
            max_offset_nm=0.5,
        )
        key_a = ("irradiated", 1, "DF", "Image A")
        key_b = ("irradiated", 1, "DF", "Image B")
        self.assertLess(
            calibration.rvis_by_image_nm[key_a],
            calibration.rvis_by_image_nm[key_b],
        )

    def test_truncated_normal_preserves_requested_mean_and_width(self):
        requested_mean = 3.2
        requested_k = 0.55
        location, scale, lower = truncated_normal_parameters_from_mean_and_k(
            requested_mean,
            requested_k,
        )
        mean, variance = truncnorm.stats(
            a=lower,
            b=np.inf,
            loc=location,
            scale=scale,
            moments="mv",
        )
        self.assertAlmostEqual(float(mean), requested_mean, places=11)
        self.assertAlmostEqual(
            float(np.sqrt(variance) / mean),
            requested_k,
            places=11,
        )
        logpdf = loop_size_logpdf(
            np.array([-1.0, 1.0, 3.0]),
            requested_mean,
            requested_k,
            "truncated_normal",
        )
        self.assertEqual(logpdf[0], -np.inf)
        self.assertTrue(np.all(np.isfinite(logpdf[1:])))

    def test_faulted_family_is_positive_normal_in_df_and_bf(self):
        theta = {
            "faulted_distribution_by_mode": {
                "DF": "zero_truncated_normal",
                "BF": "zero_truncated_normal",
            }
        }
        self.assertEqual(
            faulted_distribution_family(theta, "DF"),
            "zero_truncated_normal",
        )
        self.assertEqual(
            faulted_distribution_family(theta, "BF"),
            "zero_truncated_normal",
        )

        center = 3.0
        k = 1.1
        values = np.array([-1.0, 1.0, 3.0, 6.0])
        df_logpdf = loop_size_logpdf(
            values,
            center,
            k,
            faulted_distribution_family(theta, "DF"),
        )
        bf_logpdf = loop_size_logpdf(
            values,
            center,
            k,
            faulted_distribution_family(theta, "BF"),
        )
        self.assertEqual(df_logpdf[0], -np.inf)
        self.assertEqual(bf_logpdf[0], -np.inf)
        self.assertTrue(np.all(np.isfinite(df_logpdf[1:])))
        self.assertTrue(np.all(np.isfinite(bf_logpdf[1:])))
        np.testing.assert_allclose(df_logpdf[1:], bf_logpdf[1:])

        defaults = FitConfig()
        self.assertEqual(
            defaults.faulted_distribution_df,
            "zero_truncated_normal",
        )
        self.assertEqual(
            defaults.faulted_distribution_bf,
            "zero_truncated_normal",
        )
        self.assertEqual(
            defaults.image_visibility_rvis_overrides_nm[
                ("irradiated", 2, "DF", "Image 1135")
            ],
            1.75,
        )

    def test_paper_normal_is_not_renormalized_at_zero(self):
        center = 3.0
        k = 1.1
        location, scale, lower = positive_centered_normal_parameters(
            center,
            k,
        )
        self.assertAlmostEqual(location, center)
        self.assertAlmostEqual(scale, 3.3)
        self.assertAlmostEqual(lower, -1.0 / k)
        logpdf = loop_size_logpdf(
            np.array([-1.0, 2.0, 3.0, 4.0]),
            center,
            k,
            "normal",
        )
        self.assertEqual(logpdf[0], -np.inf)
        self.assertAlmostEqual(logpdf[1], logpdf[3], places=12)
        self.assertGreater(logpdf[2], logpdf[1])

        positive_fraction = visible_fraction_of_distribution(
            center,
            k,
            "normal",
            "DF",
            {},
        )
        self.assertAlmostEqual(
            positive_fraction,
            norm.sf(0.0, loc=center, scale=k * center),
            places=4,
        )

    def test_deprecated_bf_efficiency_does_not_change_fixed_visibility(self):
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
            prediction["Cf"] + 0.5 * prediction["Cp"],
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

        image_specific_loss, _ = image_count_deviance(
            group,
            predicted_density={"A": 1.0e16, "B": 1.0e16},
            overdispersion_floor=0.04,
        )
        self.assertAlmostEqual(image_specific_loss, matched_loss, places=12)

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
        expected_loss = coalescence_number_loss(params["Pfcs"], rf, cf)
        self.assertEqual(derivative[2], 0.0)
        self.assertAlmostEqual(derivative[4], -expected_loss)

        # Differentiate R = sqrt(Omega0*N/(pi*b*C)).  At fixed inventory,
        # dR/dt = -R/(2*C) dC/dt for conserved loop inventory.
        radius_rate = -rf * derivative[4] / (2.0 * cf)
        expected_radius_rate = 0.5 * rf * expected_loss / cf
        self.assertAlmostEqual(radius_rate, expected_radius_rate)
        self.assertGreater(radius_rate, 0.0)

    def test_perfect_coalescence_increases_radius_at_fixed_content(self):
        rp = 2.0e-7
        cp = 4.0e16
        state = np.array([
            0.0,
            0.0,
            loop_content_from_radius(1.0e-7, 8.0e16, self.material.b, self.material.Omega0),
            loop_content_from_radius(rp, cp, self.material.b, self.material.Omega0),
            8.0e16,
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
            "Di": 0.0,
            "Dv": 0.0,
            "Puf": 0.0,
            "Pcs": 3.0e-18,
            "Pfcs": 0.0,
            "enable_vacancy_extension": False,
            "enable_surface_sink": False,
        }
        derivative = rhs(0.0, state, params)
        expected_loss = coalescence_number_loss(params["Pcs"], rp, cp)
        self.assertEqual(derivative[3], 0.0)
        self.assertAlmostEqual(derivative[5], -expected_loss)

        radius_rate = -rp * derivative[5] / (2.0 * cp)
        expected_radius_rate = 0.5 * rp * expected_loss / cp
        self.assertAlmostEqual(radius_rate, expected_radius_rate)
        self.assertGreater(radius_rate, 0.0)

    def test_coalescence_loss_rejects_nonphysical_inputs(self):
        with self.assertRaises(ValueError):
            coalescence_number_loss(-1.0e-18, 1.0e-7, 1.0e16)
        with self.assertRaises(ValueError):
            coalescence_number_loss(1.0e-18, -1.0e-7, 1.0e16)
        with self.assertRaises(ValueError):
            coalescence_number_loss(1.0e-18, 1.0e-7, -1.0e16)

    def test_interaction_driven_coalescence_matches_publication_exponents(self):
        coefficient = 2.5e-22
        radius = 1.4e-7
        density = 3.2e16
        expected_inverse_lifetime = (
            coefficient
            * radius**COALESCENCE_RADIUS_EXPONENT
            * density**COALESCENCE_LIFETIME_DENSITY_EXPONENT
        )
        inverse_lifetime = coalescence_inverse_lifetime(
            coefficient,
            radius,
            density,
        )
        loss = coalescence_number_loss(coefficient, radius, density)
        self.assertAlmostEqual(inverse_lifetime, expected_inverse_lifetime)
        self.assertAlmostEqual(loss, expected_inverse_lifetime * density)

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
        self.assertNotIn("eta_BF,f", report)
        self.assertIn("DERIVED EVENT VALUES", report)


if __name__ == "__main__":
    unittest.main()
