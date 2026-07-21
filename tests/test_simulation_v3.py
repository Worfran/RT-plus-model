from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rtplus.config import DataConfig, EVENT_SERIES, FitConfig, MaterialConstants, ObservationConfig
from rtplus.data_loader import load_all_loop_data
from rtplus.initial_conditions import fitted_initial_state, fitted_initial_states
from rtplus.objective import total_objective
from rtplus.observables import predicted_mean_radii_nm, predicted_observed_number_density
from rtplus.ode import rhs
from rtplus.parameters import build_theta0_and_bounds, get_parameter_temperatures, parameter_specs, unpack_theta
from rtplus.physics import (
    compute_radius,
    lognormal_mean_radius_from_rms,
    lognormal_rms_radius_from_mean,
    loop_content_from_radius,
    loop_flux,
)
from rtplus.simulation import simulate_all_series


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
            lognormal_mean_radius_from_rms(rf, self.theta["k_f"]) * 1e7,
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


if __name__ == "__main__":
    unittest.main()
