import unittest

import numpy as np

from fib_sem_measurement_tool.core.profile_markers import collect_profile_edge_markers
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
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

    def test_horizontal_cd_uses_first_valid_edges_instead_of_strongest_texture(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 139, 19),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=30.0,
        )
        image = np.full((20, 140), 20, dtype=np.uint8)
        image[:, 20:120] = 80
        image[:, 50:90] = 230

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 100.0, delta=1.0)
        _, left_x, right_x = result.boundary_pairs[0]
        self.assertAlmostEqual(left_x, 19.5, delta=0.01)
        self.assertAlmostEqual(right_x, 119.5, delta=0.01)

    def test_explicit_center_to_outside_scan_mode_measures_inner_boundaries(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 139, 19),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=30.0,
            edge_scan_mode="center_to_outside",
        )
        image = np.full((20, 140), 20, dtype=np.uint8)
        image[:, 20:120] = 80
        image[:, 50:90] = 230

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 40.0, delta=1.0)
        _, left_x, right_x = result.boundary_pairs[0]
        self.assertAlmostEqual(left_x, 49.5, delta=0.01)
        self.assertAlmostEqual(right_x, 89.5, delta=0.01)

    def test_profile_normalization_can_recover_low_contrast_edges(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 119, 19),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=30.0,
            normalize_grayscale_profiles=True,
        )
        image = np.full((20, 120), 80, dtype=np.uint8)
        image[:, 30:90] = 98

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 60.0, delta=1.0)
        self.assertEqual(result.status, "OK")

    def test_horizontal_cd_ignores_single_column_noise_before_real_boundary(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 139, 19),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=30.0,
        )
        image = np.full((20, 140), 20, dtype=np.uint8)
        image[:, 5] = 120
        image[:, 20:120] = 80

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertIsNotNone(result.selected_px)
        self.assertAlmostEqual(result.selected_px, 100.0, delta=1.0)
        _, left_x, right_x = result.boundary_pairs[0]
        self.assertAlmostEqual(left_x, 19.5, delta=0.01)
        self.assertAlmostEqual(right_x, 119.5, delta=0.01)

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

        np.testing.assert_allclose(result.values_px, [4.0, 4.0, 4.6666665, 5.3333335, 6.0], rtol=1e-6)
        self.assertAlmostEqual(result.selected_px, 4.8, delta=0.01)
        self.assertEqual(result.valid_count, 5)

    def test_distance_both_limits_thk_scan_to_cd_interior(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 119, 99),
            measurement_type="distance_both",
            minimum_grayscale_delta=30.0,
        )
        image = np.full((100, 120), 20, dtype=np.uint8)
        image[30:70, 40:80] = 200
        image[5:15, 0:25] = 160
        image[85:95, 0:25] = 160

        result = run_measurement(image, settings)

        self.assertIsNotNone(result.horizontal_cd)
        self.assertIsNotNone(result.vertical_thk)
        self.assertAlmostEqual(result.horizontal_cd.selected_px, 40.0, delta=1.0)
        self.assertAlmostEqual(result.vertical_thk.selected_px, 40.0, delta=1.0)
        self.assertLess(result.vertical_thk.scanned_line_count, 120)
        self.assertGreaterEqual(min(pair.scan_index for pair in result.vertical_thk.selected_pairs), 43)
        self.assertLessEqual(max(pair.scan_index for pair in result.vertical_thk.selected_pairs), 76)

    def test_distance_both_respects_measure_direction_setting(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 119, 99),
            measurement_type="distance_both",
            measure_direction="vertical",
            minimum_grayscale_delta=30.0,
        )
        image = np.full((100, 120), 20, dtype=np.uint8)
        image[30:70, 40:80] = 200

        result = run_measurement(image, settings)

        self.assertIsNone(result.horizontal_cd)
        self.assertIsNotNone(result.vertical_thk)
        self.assertAlmostEqual(result.vertical_thk.selected_px, 40.0, delta=1.0)

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

    def test_boundary_angle_filter_removes_curved_cd_sections(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 119, 79),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=30.0,
            filter_cd_thk_by_boundary_angle=True,
            max_cd_thk_boundary_angle_deg=12.0,
        )
        image = np.full((80, 120), 20, dtype=np.uint8)
        for y in range(80):
            if 25 <= y <= 54:
                left, right = 30, 90
            else:
                offset = int(round(abs(y - 40) * 0.8))
                left, right = 30 + offset, 90 - offset
            if right > left:
                image[y, left:right] = 200

        filtered = measure_horizontal_cd(image, settings.roi, settings)
        unfiltered_settings = settings.clone()
        unfiltered_settings.filter_cd_thk_by_boundary_angle = False
        unfiltered = measure_horizontal_cd(image, settings.roi, unfiltered_settings)

        self.assertIsNotNone(filtered.selected_px)
        self.assertLess(filtered.valid_count, unfiltered.valid_count)
        self.assertAlmostEqual(filtered.selected_px, 60.0, delta=2.0)
        self.assertIn("angle filter removed", filtered.warning_message)

    def test_ellipse_cd_measurement_type_is_no_longer_supported(self) -> None:
        settings = MeasurementSettings(
            roi=(20, 20, 140, 120),
            measurement_type="ellipse_cd",
            minimum_grayscale_delta=25.0,
        )
        image = np.full((150, 170), 20, dtype=np.uint8)

        result = run_measurement(image, settings)

        self.assertEqual(result.status, "Fail")
        self.assertIsNone(result.ellipse_cd)
        self.assertIn("Unsupported measurement type", result.warning_message)


if __name__ == "__main__":
    unittest.main()
