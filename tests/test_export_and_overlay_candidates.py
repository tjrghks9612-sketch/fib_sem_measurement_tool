import unittest

import numpy as np

from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.core.overlay import draw_overlay
from fib_sem_measurement_tool.export.csv_exporter import CSV_COLUMNS, make_result_row
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MeasurementSettings


class ExportAndOverlayCandidateTest(unittest.TestCase):
    def _measured_item(self):
        settings = MeasurementSettings(
            roi=(2, 1, 12, 10),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=20.0,
        )
        image = np.zeros((14, 16, 3), dtype=np.uint8)
        image[1:11, 4:10, :] = 120
        result = run_measurement(image, settings)
        item = ImageItem(image_path="sample.png", file_name="sample.png", image_size=(16, 14), result=result)
        return image, item, settings

    def test_csv_includes_candidate_debug_fields(self) -> None:
        _image, item, settings = self._measured_item()

        row = make_result_row(item, settings)

        for column in [
            "raw_edge_count",
            "candidate_coverage",
            "selected_point_count",
            "pair_candidate_count",
            "horizontal_cd_raw_edge_count",
            "horizontal_cd_scanline_coverage",
            "horizontal_cd_selected_point_count",
        ]:
            self.assertIn(column, CSV_COLUMNS)
            self.assertIn(column, row)
        self.assertGreater(row["raw_edge_count"], 0)
        self.assertGreater(row["horizontal_cd_raw_edge_count"], 0)

    def test_csv_includes_ellipse_cd_fields(self) -> None:
        settings = MeasurementSettings(
            roi=(20, 20, 140, 120),
            measurement_type="ellipse_cd",
            minimum_grayscale_delta=25.0,
        )
        image = np.full((150, 170), 20, dtype=np.uint8)
        yy, xx = np.indices(image.shape[:2])
        image[((xx - 80) / 42.0) ** 2 + ((yy - 70) / 28.0) ** 2 <= 1.0] = 190
        result = run_measurement(image, settings)
        item = ImageItem(image_path="ellipse.png", file_name="ellipse.png", image_size=(170, 150), result=result)

        row = make_result_row(item, settings)

        for column in [
            "ellipse_cd_ray_attempt_count",
            "ellipse_cd_valid_point_count",
            "ellipse_cd_outlier_count",
            "ellipse_cd_center_x",
            "ellipse_cd_center_y",
            "ellipse_cd_major_axis_px",
            "ellipse_cd_minor_axis_px",
            "ellipse_cd_angle_deg",
            "ellipse_cd_horizontal_diameter_px",
            "ellipse_cd_vertical_diameter_px",
            "ellipse_cd_horizontal_diameter",
            "ellipse_cd_vertical_diameter",
            "ellipse_cd_confidence",
            "ellipse_cd_status",
            "ellipse_cd_warning_message",
        ]:
            self.assertIn(column, CSV_COLUMNS)
            self.assertIn(column, row)
        self.assertGreater(row["ellipse_cd_valid_point_count"], 0)

    def test_raw_candidate_overlay_toggle_does_not_render_debug_points(self) -> None:
        image, item, settings = self._measured_item()
        self.assertIsNotNone(item.result)

        settings.show_raw_candidates = True
        with_raw = draw_overlay(image, settings.roi, item.result, settings, show_overlay=True)
        settings.show_raw_candidates = False
        without_raw = draw_overlay(image, settings.roi, item.result, settings, show_overlay=True)

        self.assertEqual(int(np.sum(np.abs(with_raw.astype(np.int16) - without_raw.astype(np.int16)))), 0)


if __name__ == "__main__":
    unittest.main()
