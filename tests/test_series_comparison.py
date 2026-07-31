from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rtplus.config import DataConfig
from rtplus.data_loader import load_all_loop_data
from rtplus.series_comparison import (
    ImageSample,
    benjamini_hochberg,
    compare_image_samples_exact,
    compare_matched_pristine_irradiated,
)


def _sample(series_id, image_id, volume_nm3, diameters_nm):
    return ImageSample(
        series_id=series_id,
        source_file=f"{series_id}.csv",
        image_id=image_id,
        volume_nm3=float(volume_nm3),
        diameters_nm=np.asarray(diameters_nm, dtype=float),
    )


class SeriesComparisonTests(unittest.TestCase):
    def test_exact_comparison_uses_complete_images(self):
        pristine = [
            _sample("pristine", "P1", 100.0, [1.0, 1.2]),
            _sample("pristine", "P2", 200.0, [1.1, 1.3, 1.4]),
        ]
        irradiated = [
            _sample("irradiated", "I1", 100.0, [2.5, 2.7, 2.8, 3.0]),
            _sample("irradiated", "I2", 200.0, [2.4, 2.9, 3.1, 3.2, 3.4]),
        ]
        result = compare_image_samples_exact(pristine, irradiated)
        expected_pristine_density = np.mean([2.0 / 100.0, 3.0 / 200.0])
        expected_irradiated_density = np.mean([4.0 / 100.0, 5.0 / 200.0])
        self.assertAlmostEqual(
            result["density_ratio"],
            expected_irradiated_density / expected_pristine_density,
        )
        self.assertEqual(result["n_assignments"], 6)
        self.assertGreater(result["wasserstein_nm"], 1.0)
        self.assertGreaterEqual(result["joint_p_exact"], result["joint_min_p"])

    def test_benjamini_hochberg_preserves_original_order(self):
        adjusted = benjamini_hochberg([0.20, 0.01, 0.04])
        np.testing.assert_allclose(adjusted, [0.20, 0.03, 0.06])

    def test_real_data_find_four_matched_comparisons(self):
        loop_data = load_all_loop_data(DataConfig(), project_root=ROOT)
        results = compare_matched_pristine_irradiated(loop_data)
        self.assertEqual(len(results), 4)
        self.assertEqual(set(results["temperature_C"]), {25.0, 900.0})
        self.assertEqual(set(results["mode"]), {"BF", "DF"})
        rt_df = results[
            np.isclose(results["temperature_C"], 25.0)
            & (results["mode"] == "DF")
        ].iloc[0]
        self.assertEqual(rt_df["n_images_pristine"], 1)
        self.assertEqual(rt_df["n_images_irradiated"], 3)
        self.assertGreaterEqual(rt_df["joint_min_p"], 0.25)


if __name__ == "__main__":
    unittest.main()
