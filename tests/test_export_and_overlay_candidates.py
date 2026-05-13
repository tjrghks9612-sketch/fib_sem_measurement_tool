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
