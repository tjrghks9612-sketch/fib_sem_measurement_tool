import unittest
from pathlib import Path

import cv2
import numpy as np

from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.models.settings import MeasurementSettings


SAMPLE_DIR = Path(r"C:\Users\admin\Downloads\fib_sem_raw_samples_CD_THK_Taper_6pcs")


def load_sample(name: str) -> np.ndarray:
    path = SAMPLE_DIR / name
    if not path.exists():
        raise unittest.SkipTest(f"sample image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise unittest.SkipTest(f"sample image could not be read: {path}")
    return image


def settings_for(measurement_type: str, roi: tuple[int, int, int, int]) -> MeasurementSettings:
    return MeasurementSettings(
        roi=roi,
        measurement_type=measurement_type,
        minimum_grayscale_delta=20.0,
    )


class SampleImageMeasurementTest(unittest.TestCase):
    def test_cd_samples_measure_central_trench_width(self) -> None:
        cases = [
            ("CD_sample_01_raw_trench.png", (320, 150, 710, 610), 292.0, 304.0),
            ("CD_sample_02_raw_pillar_gap.png", (300, 150, 720, 610), 260.0, 285.0),
        ]

        for name, roi, low, high in cases:
            with self.subTest(name=name):
                result = run_measurement(load_sample(name), settings_for("distance_horizontal", roi))

                self.assertIsNotNone(result.horizontal_cd)
                self.assertEqual(result.horizontal_cd.status, "OK")
                self.assertGreaterEqual(result.horizontal_cd.confidence, 95.0)
                self.assertGreaterEqual(result.horizontal_cd.selected_px, low)
                self.assertLessEqual(result.horizontal_cd.selected_px, high)

    def test_thk_samples_pick_intended_horizontal_layer_boundaries(self) -> None:
        cases = [
            ("THK_sample_01_raw_film_stack.png", (0, 245, 1023, 505), 158.0, 173.0, 260.0, 275.0, 425.0, 440.0),
            ("THK_sample_02_raw_thin_layer.png", (0, 250, 1023, 500), 88.0, 100.0, 260.0, 276.0, 352.0, 370.0),
        ]

        for name, roi, low, high, top_low, top_high, bottom_low, bottom_high in cases:
            with self.subTest(name=name):
                result = run_measurement(load_sample(name), settings_for("distance_vertical", roi))

                self.assertIsNotNone(result.vertical_thk)
                thk = result.vertical_thk
                self.assertEqual(thk.status, "OK")
                self.assertGreaterEqual(thk.confidence, 95.0)
                self.assertGreaterEqual(thk.selected_px, low)
                self.assertLessEqual(thk.selected_px, high)
                self.assertGreaterEqual(float(np.median([pair.first.image_y for pair in thk.selected_pairs])), top_low)
                self.assertLessEqual(float(np.median([pair.first.image_y for pair in thk.selected_pairs])), top_high)
                self.assertGreaterEqual(float(np.median([pair.second.image_y for pair in thk.selected_pairs])), bottom_low)
                self.assertLessEqual(float(np.median([pair.second.image_y for pair in thk.selected_pairs])), bottom_high)

    def test_taper_samples_fit_wall_edges_near_target_height(self) -> None:
        single = settings_for("taper_single", (260, 145, 600, 610))
        single.taper_side = "left"
        single_result = run_measurement(load_sample("Taper_sample_01_raw_single_side.png"), single)
        self.assertIsNotNone(single_result.left_taper)
        self.assertEqual(single_result.left_taper.status, "OK")
        self.assertGreaterEqual(single_result.left_taper.angle_vertical, 1.5)
        self.assertLessEqual(single_result.left_taper.angle_vertical, 5.0)
        self.assertGreaterEqual(single_result.left_taper.selected_point_count, 100)

        double = settings_for("taper_double", (260, 145, 760, 610))
        double_result = run_measurement(load_sample("Taper_sample_02_raw_double_side.png"), double)
        self.assertIsNotNone(double_result.left_taper)
        self.assertIsNotNone(double_result.right_taper)
        self.assertEqual(double_result.left_taper.status, "OK")
        self.assertEqual(double_result.right_taper.status, "OK")
        self.assertGreaterEqual(double_result.left_taper.angle_vertical, 3.5)
        self.assertLessEqual(double_result.left_taper.angle_vertical, 7.0)
        self.assertGreaterEqual(double_result.right_taper.angle_vertical, 7.0)
        self.assertLessEqual(double_result.right_taper.angle_vertical, 12.0)
        self.assertGreaterEqual(double_result.left_taper.selected_point_count, 100)
        self.assertGreaterEqual(double_result.right_taper.selected_point_count, 100)

    def test_sample_measurements_remain_stable_with_normalized_denoised_profiles(self) -> None:
        cd = settings_for("distance_horizontal", (300, 150, 720, 610))
        cd.normalize_grayscale_profiles = True
        cd.denoise_grayscale_profiles = True
        cd_result = run_measurement(load_sample("CD_sample_02_raw_pillar_gap.png"), cd)
        self.assertIsNotNone(cd_result.horizontal_cd)
        self.assertEqual(cd_result.horizontal_cd.status, "OK")
        self.assertGreaterEqual(cd_result.horizontal_cd.selected_px, 260.0)
        self.assertLessEqual(cd_result.horizontal_cd.selected_px, 285.0)

        thk = settings_for("distance_vertical", (0, 250, 1023, 500))
        thk.normalize_grayscale_profiles = True
        thk.denoise_grayscale_profiles = True
        thk_result = run_measurement(load_sample("THK_sample_02_raw_thin_layer.png"), thk)
        self.assertIsNotNone(thk_result.vertical_thk)
        self.assertEqual(thk_result.vertical_thk.status, "OK")
        self.assertGreaterEqual(thk_result.vertical_thk.selected_px, 88.0)
        self.assertLessEqual(thk_result.vertical_thk.selected_px, 100.0)

        taper = settings_for("taper_double", (260, 145, 760, 610))
        taper.normalize_grayscale_profiles = True
        taper.denoise_grayscale_profiles = True
        taper_result = run_measurement(load_sample("Taper_sample_02_raw_double_side.png"), taper)
        self.assertIsNotNone(taper_result.left_taper)
        self.assertIsNotNone(taper_result.right_taper)
        self.assertEqual(taper_result.left_taper.status, "OK")
        self.assertEqual(taper_result.right_taper.status, "OK")
        self.assertGreaterEqual(taper_result.left_taper.angle_vertical, 3.5)
        self.assertLessEqual(taper_result.left_taper.angle_vertical, 7.0)
        self.assertGreaterEqual(taper_result.right_taper.angle_vertical, 7.0)
        self.assertLessEqual(taper_result.right_taper.angle_vertical, 12.0)
