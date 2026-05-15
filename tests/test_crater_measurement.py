import os
import unittest

import cv2
import numpy as np

from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.export.csv_exporter import CSV_COLUMNS, make_result_row
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def synthetic_crater() -> np.ndarray:
    h, w = 260, 420
    image = np.full((h, w), 120, dtype=np.uint8)
    baseline = 178
    center = w // 2
    half_width = 120
    height = 70
    xs = np.arange(w)
    profile = np.full(w, baseline, dtype=np.float32)
    inside = np.abs(xs - center) <= half_width
    dome = np.sqrt(np.maximum(0.0, 1.0 - ((xs[inside] - center) / half_width) ** 2))
    profile[inside] = baseline - height * dome
    for x, y in enumerate(profile.astype(int)):
        image[y:, x] = 42
        image[max(0, y - 2) : min(h, y + 3), x] = 210
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


class CraterMeasurementTest(unittest.TestCase):
    def test_crater_mode_measures_cd_thk_and_taper_on_profile(self) -> None:
        image = synthetic_crater()
        settings = MeasurementSettings(roi=(12, 35, 407, 225), measurement_type="crater", minimum_grayscale_delta=35)
        result = run_measurement(image, settings)

        self.assertIsNotNone(result.crater)
        crater = result.crater
        self.assertIn(result.status, {"OK", "Check", "Review Needed"})
        self.assertGreater(crater.cd_px, 190)
        self.assertLess(crater.cd_px, 270)
        self.assertGreater(crater.thk_px, 50)
        self.assertLess(crater.thk_px, 90)
        self.assertIsNotNone(crater.left_taper_angle_horizontal)
        self.assertIsNotNone(crater.right_taper_angle_horizontal)
        self.assertEqual(crater.taper_height_percent, 15.0)
        self.assertIsNotNone(crater.left_taper_measure_y)
        self.assertIsNotNone(crater.right_taper_measure_y)
        self.assertGreater(crater.top_profile_valid_count, 150)

    def test_crater_csv_columns_are_added_without_renaming_existing_columns(self) -> None:
        image = synthetic_crater()
        settings = MeasurementSettings(roi=(12, 35, 407, 225), measurement_type="crater", minimum_grayscale_delta=35)
        result = run_measurement(image, settings)
        item = ImageItem(image_path="synthetic_crater.png", file_name="synthetic_crater.png", image_size=(image.shape[1], image.shape[0]), result=result)
        row = make_result_row(item, settings)

        self.assertIn("horizontal_cd_selected_px", CSV_COLUMNS)
        self.assertIn("hole_cd_horizontal_px", CSV_COLUMNS)
        self.assertIn("crater_cd_px", CSV_COLUMNS)
        self.assertIn("crater_thk_px", CSV_COLUMNS)
        self.assertIn("crater_taper_height_percent", CSV_COLUMNS)
        self.assertGreater(row["crater_cd_px"], 0)
        self.assertGreater(row["crater_thk_px"], 0)
        self.assertEqual(row["crater_taper_height_percent"], 15.0)

    def test_crater_taper_height_percent_moves_measure_marker(self) -> None:
        image = synthetic_crater()
        roi = (12, 35, 407, 225)
        low = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="crater", minimum_grayscale_delta=35, crater_taper_height_percent=15))
        high = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="crater", minimum_grayscale_delta=35, crater_taper_height_percent=55))

        self.assertIsNotNone(low.crater)
        self.assertIsNotNone(high.crater)
        self.assertEqual(low.crater.taper_height_percent, 15.0)
        self.assertEqual(high.crater.taper_height_percent, 55.0)
        self.assertNotEqual(low.crater.left_taper_measure_y, high.crater.left_taper_measure_y)
        self.assertLess(high.crater.left_taper_measure_y, low.crater.left_taper_measure_y)

    def test_non_crater_rows_leave_crater_columns_blank(self) -> None:
        item = ImageItem(image_path="unmeasured.png", file_name="unmeasured.png", image_size=(1, 1), result=None)
        row = make_result_row(item, MeasurementSettings(measurement_type="distance_both"))

        self.assertEqual(row["crater_cd_px"], "")
        self.assertEqual(row["crater_status"], "")

    def test_reference_crater_samples_measure_when_available(self) -> None:
        paths = [
            "C:/Users/admin/Downloads/fib_sem_raw_samples/Crater 1.png",
            "C:/Users/admin/Downloads/fib_sem_raw_samples/Crater 2.png",
        ]
        for path in paths:
            if not os.path.exists(path):
                self.skipTest(f"missing reference image: {path}")
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            roi = (0, int(h * 0.22), w - 1, int(h * 0.75))
            result = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="crater", minimum_grayscale_delta=55))

            self.assertIsNotNone(result.crater)
            self.assertNotEqual(result.status, "Fail")
            self.assertGreater(result.crater.cd_px, 100)
            self.assertGreater(result.crater.thk_px, 20)


if __name__ == "__main__":
    unittest.main()
