import unittest

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    detect_profile_candidates,
    prepare_display_profile_signal,
    scan_pair_candidates,
)
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_taper import measure_single_taper
from fib_sem_measurement_tool.models.result import EdgeScanResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def make_settings() -> MeasurementSettings:
    return MeasurementSettings(roi=(20, 20, 179, 139), measurement_type="distance_both")


class GrayscaleLineScanTest(unittest.TestCase):
    def test_detects_all_raw_grayscale_changes_without_grouping(self) -> None:
        settings = make_settings()
        profile = np.full(140, 20, dtype=np.uint8)
        profile[10:30] = 80
        profile[50:90] = 210
        profile[110:130] = 90

        candidates = detect_profile_candidates(
            profile,
            settings,
            scan_axis="horizontal",
            scan_index=0,
            local_scan_index=0,
            roi_origin=(0, 0),
        )
        pair = max(
            scan_pair_candidates(EdgeScanResult("horizontal", (0, 0, profile.size - 1, 0), 1, candidates)),
            key=lambda candidate: candidate.distance,
        )

        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair.first.position, 9.5, delta=0.01)
        self.assertAlmostEqual(pair.second.position, 129.5, delta=0.01)

    def test_minimum_grayscale_delta_blocks_weak_changes(self) -> None:
        settings = make_settings()
        settings.minimum_grayscale_delta = 80.0
        weak = np.full(120, 20, dtype=np.uint8)
        weak[40:80] = 70
        strong = np.full(120, 20, dtype=np.uint8)
        strong[40:80] = 130

        self.assertEqual(
            detect_profile_candidates(weak, settings, "horizontal", 0, 0, (0, 0)),
            [],
        )
        self.assertGreater(
            len(detect_profile_candidates(strong, settings, "horizontal", 0, 0, (0, 0))),
            0,
        )

    def test_display_profile_signal_applies_normalization_and_smoothing(self) -> None:
        settings = make_settings()
        settings.normalize_grayscale_profiles = False
        settings.denoise_grayscale_profiles = True
        settings.profile_denoise_window = 9
        settings.profile_denoise_range_sigma = 18.0
        profile = np.asarray([80, 84, 76, 83, 78, 82, 80, 160, 162, 158, 161, 159], dtype=np.float32)

        display = prepare_display_profile_signal(profile, "horizontal", settings)

        self.assertEqual(display.shape, profile.shape)
        self.assertLess(float(np.std(display[:7])), float(np.std(profile[:7])))
        self.assertGreater(float(display[7] - display[6]), 50.0)

    def test_cd_and_thk_use_same_line_scanner(self) -> None:
        settings = make_settings()
        image = np.full((160, 220), 20, dtype=np.uint8)
        image[40:120, 70:150] = 210

        horizontal = measure_horizontal_cd(image, settings.roi, settings)
        vertical = measure_vertical_thk(image, settings.roi, settings)

        self.assertAlmostEqual(horizontal.selected_px, 80.0, delta=2.0)
        self.assertAlmostEqual(vertical.selected_px, 80.0, delta=2.0)
        self.assertNotEqual(horizontal.status, "Fail")
        self.assertNotEqual(vertical.status, "Fail")

    def test_taper_uses_scanned_side_points(self) -> None:
        settings = make_settings()
        settings.measurement_type = "taper_single"
        image = np.full((160, 220), 20, dtype=np.uint8)
        for y in range(35, 130):
            left = int(round(65 + 0.20 * (y - 35)))
            right = int(round(155 - 0.08 * (y - 35)))
            image[y, left:right] = 210

        result = measure_single_taper(image, settings.roi, "left", settings)

        self.assertIsNotNone(result.left_taper)
        self.assertNotEqual(result.left_taper.status, "Fail")
        self.assertGreaterEqual(result.left_taper.valid_point_count, 5)
        self.assertAlmostEqual(result.left_taper.angle_vertical, 11.3, delta=3.0)


if __name__ == "__main__":
    unittest.main()
