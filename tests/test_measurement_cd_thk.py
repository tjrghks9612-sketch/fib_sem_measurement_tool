import unittest

import numpy as np

from fib_sem_measurement_tool.core.profile_markers import collect_profile_edge_markers
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def make_settings() -> MeasurementSettings:
    return MeasurementSettings(roi=(20, 20, 179, 139), measurement_type="distance_both")


def make_step_image() -> np.ndarray:
    image = np.full((160, 220), 20, dtype=np.uint8)
    image[40:120, 70:150] = 210
    return image


class CdThkMeasurementTest(unittest.TestCase):
    def test_horizontal_cd_uses_grayscale_edges_inside_roi(self) -> None:
        settings = make_settings()
        result = measure_horizontal_cd(make_step_image(), settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 80.0, delta=2.0)
        self.assertGreaterEqual(result.valid_count, 3)
        self.assertNotEqual(result.status, "Fail")

        _, left_x, right_x = result.boundary_pairs[0]
        self.assertAlmostEqual(left_x, 70.0, delta=2.0)
        self.assertAlmostEqual(right_x, 149.0, delta=2.0)

    def test_vertical_thk_uses_grayscale_edges_inside_roi(self) -> None:
        settings = make_settings()
        result = measure_vertical_thk(make_step_image(), settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 80.0, delta=2.0)
        self.assertGreaterEqual(result.valid_count, 3)
        self.assertNotEqual(result.status, "Fail")

        _, top_y, bottom_y = result.boundary_pairs[0]
        self.assertAlmostEqual(top_y, 40.0, delta=2.0)
        self.assertAlmostEqual(bottom_y, 119.0, delta=2.0)

    def test_horizontal_cd_keeps_right_boundary_candidates_when_roi_has_multiple_edges(self) -> None:
        settings = make_settings()
        image = np.zeros((160, 220), dtype=np.uint8)
        image[:, 70:76] = 220
        image[:, 100:106] = 140
        image[:, 150:159] = 220

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertGreaterEqual(result.valid_count, 3)
        self.assertNotEqual(result.status, "Fail")
        _, left_x, right_x = result.boundary_pairs[0]
        self.assertLess(left_x, right_x)

    def test_vertical_thk_measures_when_horizontal_edges_cover_only_part_of_roi_width(self) -> None:
        settings = make_settings()
        image = np.full((160, 220), 20, dtype=np.uint8)
        image[40:120, 90:130] = 210

        result = measure_vertical_thk(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 80.0, delta=2.0)
        self.assertGreaterEqual(result.valid_count, 3)
        self.assertNotEqual(result.status, "Fail")

    def test_vertical_thk_uses_same_scanline_side_pair_selection_as_cd(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 4, 9),
            measurement_type="distance_vertical",
            minimum_grayscale_delta=20.0,
        )
        image = np.full((10, 5), 90, dtype=np.uint8)
        image[2:, :] = 30
        image[6:, 0:3] = 95
        image[8:, 3:5] = 140

        result = measure_vertical_thk(image, settings.roi, settings)

        self.assertEqual(result.values_px, [4.0, 4.0, 4.0, 6.0, 6.0])
        self.assertAlmostEqual(result.selected_px, 4.8, delta=0.01)
        self.assertEqual(result.valid_count, 5)

    def test_vertical_thk_matches_horizontal_cd_on_transposed_image(self) -> None:
        image = np.full((90, 130), 25, dtype=np.uint8)
        image[20:72, 35:104] = 185
        image[35:60, 62:65] = 70
        settings = MeasurementSettings(
            roi=(10, 8, 118, 82),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=20.0,
        )
        transposed_settings = MeasurementSettings(
            roi=(8, 10, 82, 118),
            measurement_type="distance_vertical",
            minimum_grayscale_delta=20.0,
        )

        cd = measure_horizontal_cd(image, settings.roi, settings)
        thk = measure_vertical_thk(image.T, transposed_settings.roi, transposed_settings)

        self.assertEqual(cd.values_px, thk.values_px)
        self.assertEqual(cd.valid_count, thk.valid_count)
        self.assertAlmostEqual(cd.selected_px, thk.selected_px, delta=0.001)

    def test_distance_confidence_uses_selected_pair_coverage_not_raw_edge_coverage(self) -> None:
        settings = MeasurementSettings(
            roi=(20, 20, 179, 139),
            measurement_type="distance_vertical",
            minimum_grayscale_delta=20.0,
        )
        image = np.full((160, 220), 20, dtype=np.uint8)
        image[40:, :] = 100
        image[120:, 90:130] = 210

        result = measure_vertical_thk(image, settings.roi, settings)

        self.assertEqual(result.scanline_coverage, 1.0)
        self.assertLess(result.confidence, 50.0)
        self.assertEqual(result.status, "Check")

    def test_profile_markers_fall_back_to_representative_selected_edge(self) -> None:
        settings = MeasurementSettings(
            roi=(20, 20, 179, 139),
            measurement_type="distance_vertical",
            minimum_grayscale_delta=20.0,
        )
        image = np.full((160, 220), 20, dtype=np.uint8)
        image[40:120, 90:130] = 210
        thk = measure_vertical_thk(image, settings.roi, settings)

        markers = collect_profile_edge_markers(
            MeasurementResult(measurement_type="distance_vertical", vertical_thk=thk),
            axis="vertical",
            scan_index=20,
        )

        self.assertEqual([marker.label for marker in markers], ["THK T", "THK B"])
        self.assertAlmostEqual(markers[0].position, 40.0, delta=2.0)
        self.assertAlmostEqual(markers[1].position, 119.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
