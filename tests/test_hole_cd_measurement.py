import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.export.csv_exporter import CSV_COLUMNS, make_result_row
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def synthetic_concentric_hole() -> np.ndarray:
    h, w = 240, 260
    yy, xx = np.mgrid[:h, :w]
    cx, cy = 132.0, 118.0
    radius = np.sqrt(((xx - cx) / 1.12) ** 2 + ((yy - cy) / 0.92) ** 2)
    image = np.full((h, w), 180, dtype=np.uint8)
    image[radius < 92] = 118
    image[radius < 58] = 142
    image[radius < 38] = 35
    image[((xx - 118) ** 2 + (yy - 103) ** 2) < 11 ** 2] = 80
    return np.dstack([image, image, image])


class HoleCDMeasurementTest(unittest.TestCase):
    def test_inner_and_outer_select_different_continuous_boundaries(self) -> None:
        image = synthetic_concentric_hole()
        roi = (22, 18, 238, 218)

        inner = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="hole_cd", hole_target="inner", minimum_grayscale_delta=25))
        outer = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="hole_cd", hole_target="outer", minimum_grayscale_delta=25))

        self.assertIsNotNone(inner.hole_cd)
        self.assertIsNotNone(outer.hole_cd)
        self.assertNotEqual(inner.status, "Fail")
        self.assertNotEqual(outer.status, "Fail")
        self.assertLess(inner.hole_cd.mean_radius, outer.hole_cd.mean_radius)
        self.assertLess(inner.hole_cd.horizontal_px, outer.hole_cd.horizontal_px)
        self.assertNotAlmostEqual(inner.hole_cd.horizontal_px, inner.hole_cd.vertical_px, delta=1.0)

    def test_hole_cd_csv_columns_are_added_without_renaming_existing_columns(self) -> None:
        image = synthetic_concentric_hole()
        settings = MeasurementSettings(roi=(22, 18, 238, 218), measurement_type="hole_cd", hole_target="outer", minimum_grayscale_delta=25)
        result = run_measurement(image, settings)
        item = ImageItem(image_path="hole.png", file_name="hole.png", image_size=(260, 240), result=result)

        row = make_result_row(item, settings)

        self.assertIn("horizontal_cd_selected_px", CSV_COLUMNS)
        self.assertIn("hole_cd_horizontal_px", CSV_COLUMNS)
        self.assertEqual(row["hole_target"], "outer")
        self.assertGreater(row["hole_cd_area_px"], 0)

    def test_non_hole_measurement_leaves_hole_columns_blank(self) -> None:
        settings = MeasurementSettings(roi=(0, 0, 20, 20), measurement_type="distance_horizontal")
        item = ImageItem(image_path="cd.png", file_name="cd.png", image_size=(21, 21), result=None)

        row = make_result_row(item, settings)

        self.assertEqual(row["hole_target"], "")
        self.assertEqual(row["hole_cd_horizontal_px"], "")

    def test_sample_images_produce_reviewable_hole_boundaries(self) -> None:
        sample_dir = Path(r"C:\Users\admin\Downloads\fib_sem_raw_samples")
        for name in ("Hole CD 1.png", "Hole CD 2.png"):
            path = sample_dir / name
            if not path.exists():
                continue
            image = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
            h, w = image.shape[:2]
            roi = (int(w * 0.12), int(h * 0.10), int(w * 0.88), int(h * 0.90))
            inner = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="hole_cd", hole_target="inner", minimum_grayscale_delta=30))
            outer = run_measurement(image, MeasurementSettings(roi=roi, measurement_type="hole_cd", hole_target="outer", minimum_grayscale_delta=30))

            self.assertIsNotNone(inner.hole_cd)
            self.assertIsNotNone(outer.hole_cd)
            self.assertNotEqual(inner.status, "Fail")
            self.assertNotEqual(outer.status, "Fail")
            self.assertLess(inner.hole_cd.mean_radius, outer.hole_cd.mean_radius)


if __name__ == "__main__":
    unittest.main()
