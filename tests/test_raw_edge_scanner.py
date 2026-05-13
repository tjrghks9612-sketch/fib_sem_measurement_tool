import unittest
from unittest.mock import patch

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    detect_profile_candidates,
    fill_small_gaps_nan,
    moving_average_nan,
    refine_boundary_candidates_by_line,
    scan_pair_candidates,
    scan_raw_edge_candidates,
    scan_edge_in_direction,
    select_first_valid_boundary_pairs_per_scanline,
    select_refined_side_pair_per_scanline,
    select_strongest_boundary_candidates,
)
from fib_sem_measurement_tool.models.result import RawEdgeCandidate
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.core.measurement_taper import LimitPeakBoundaryEngine
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def make_settings() -> MeasurementSettings:
    return MeasurementSettings(
        roi=(2, 1, 7, 4),
        measurement_type="distance_horizontal",
        minimum_grayscale_delta=20.0,
    )


class RawEdgeScannerTest(unittest.TestCase):
    def test_profile_candidates_use_raw_grayscale_differences(self) -> None:
        settings = make_settings()
        profile = np.asarray([10, 10, 40, 5, 5], dtype=np.uint8)

        candidates = detect_profile_candidates(
            profile,
            settings,
            scan_axis="horizontal",
            scan_index=12,
            local_scan_index=2,
            roi_origin=(10, 10),
        )

        self.assertEqual([candidate.position for candidate in candidates], [1.5, 2.5])
        self.assertEqual([candidate.sign for candidate in candidates], [1, -1])
        self.assertEqual([candidate.signed_delta for candidate in candidates], [30.0, -35.0])
        self.assertEqual([candidate.strength for candidate in candidates], [30.0, 35.0])
        self.assertEqual(candidates[0].grayscale_before, 10.0)
        self.assertEqual(candidates[0].grayscale_after, 40.0)
        self.assertEqual(candidates[0].image_x, 11.5)
        self.assertEqual(candidates[0].image_y, 12.0)

    def test_horizontal_scan_checks_every_roi_row(self) -> None:
        settings = make_settings()
        image = np.zeros((6, 10), dtype=np.uint8)
        image[1:5, 5:] = 80

        scan = scan_raw_edge_candidates(image, settings.roi, "horizontal", settings)

        self.assertEqual(scan.scanned_line_count, 4)
        self.assertEqual(sorted(scan.per_scanline_candidate_count), [1, 2, 3, 4])
        self.assertEqual(set(candidate.local_scan_index for candidate in scan.raw_edge_candidates), {0, 1, 2, 3})

    def test_vertical_scan_checks_every_roi_col(self) -> None:
        settings = make_settings()
        image = np.zeros((6, 10), dtype=np.uint8)
        image[3:, 2:8] = 80

        scan = scan_raw_edge_candidates(image, settings.roi, "vertical", settings)

        self.assertEqual(scan.scanned_line_count, 6)
        self.assertEqual(sorted(scan.per_scanline_candidate_count), [2, 3, 4, 5, 6, 7])
        self.assertEqual(set(candidate.local_scan_index for candidate in scan.raw_edge_candidates), {0, 1, 2, 3, 4, 5})

    def test_threshold_keeps_sobel_local_peaks_at_or_above_minimum_delta(self) -> None:
        settings = make_settings()
        settings.minimum_grayscale_delta = 25.0
        profile = np.asarray([10, 30, 55, 79, 105], dtype=np.uint8)

        candidates = detect_profile_candidates(
            profile,
            settings,
            scan_axis="horizontal",
            scan_index=1,
            local_scan_index=0,
            roi_origin=(0, 0),
        )

        self.assertEqual([candidate.position for candidate in candidates], [3.5])
        self.assertEqual([candidate.signed_delta for candidate in candidates], [50.0])

    def test_pair_selection_preserves_raw_candidates_and_pair_candidates(self) -> None:
        settings = make_settings()
        image = np.zeros((6, 10), dtype=np.uint8)
        image[1:5, 3:5] = 70
        image[1:5, 6:8] = 140

        result = measure_horizontal_cd(image, settings.roi, settings)

        self.assertGreater(result.raw_edge_count, 0)
        self.assertEqual(len(result.raw_edge_candidates), result.raw_edge_count)
        self.assertGreater(result.pair_candidate_count, 0)
        self.assertEqual(len(result.pair_candidates), result.pair_candidate_count)
        self.assertIsNotNone(result.selected_pair)
        self.assertGreater(result.scanline_coverage, 0.0)

    def test_vertical_measurement_preserves_raw_candidates(self) -> None:
        settings = make_settings()
        image = np.zeros((6, 10), dtype=np.uint8)
        image[2:4, 2:8] = 80

        result = measure_vertical_thk(image, settings.roi, settings)

        self.assertEqual(result.scanned_line_count, 6)
        self.assertGreater(result.raw_edge_count, 0)
        self.assertGreater(result.pair_candidate_count, 0)
        self.assertIsNotNone(result.selected_pair)

    def test_run_measurement_does_not_call_preprocess_image(self) -> None:
        settings = MeasurementSettings(
            roi=(2, 1, 12, 10),
            measurement_type="distance_horizontal",
            minimum_grayscale_delta=20.0,
        )
        image = np.zeros((14, 16, 3), dtype=np.uint8)
        image[1:11, 4:10, :] = 120

        with patch(
            "fib_sem_measurement_tool.core.measurement_runner.preprocess_image",
            side_effect=AssertionError("preprocess_image must not be used for measurement"),
            create=True,
        ):
            result = run_measurement(image, settings)

        self.assertNotEqual(result.status, "Fail")
        self.assertIsNotNone(result.horizontal_cd)
        self.assertGreater(result.horizontal_cd.raw_edge_count, 0)

    def test_run_measurement_counts_failed_submeasurement_in_overall_coverage(self) -> None:
        settings = MeasurementSettings(
            roi=(2, 1, 12, 10),
            measurement_type="distance_both",
            minimum_grayscale_delta=20.0,
        )
        image = np.zeros((14, 16, 3), dtype=np.uint8)
        image[:, 4:10, :] = 120

        result = run_measurement(image, settings)

        self.assertIsNotNone(result.horizontal_cd)
        self.assertIsNotNone(result.vertical_thk)
        self.assertNotEqual(result.horizontal_cd.status, "Fail")
        self.assertEqual(result.vertical_thk.status, "Fail")
        self.assertLess(result.overall_confidence, 80.0)
        self.assertNotEqual(result.status, "OK")

    def test_scan_pair_candidates_keeps_all_combinations_on_same_line(self) -> None:
        settings = make_settings()
        image = np.zeros((3, 8), dtype=np.uint8)
        image[:, 2:4] = 80
        image[:, 5:7] = 160

        scan = scan_raw_edge_candidates(image, (0, 0, 7, 2), "horizontal", settings)
        pairs = scan_pair_candidates(scan)

        self.assertGreater(len(pairs), scan.scanned_line_count)
        self.assertTrue(all(pair.first.position < pair.second.position for pair in pairs))

    def test_refined_pair_selection_uses_strongest_sobel_local_peak(self) -> None:
        settings = make_settings()
        image = np.asarray(
            [
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
            ],
            dtype=np.float32,
        )

        scan = scan_raw_edge_candidates(image, (0, 0, 10, 1), "horizontal", settings)
        selected = select_refined_side_pair_per_scanline(scan)

        self.assertEqual(len(selected), 2)
        self.assertEqual([pair.first.position for pair in selected], [3.5, 3.5])
        self.assertEqual([pair.second.position for pair in selected], [8.5, 8.5])

    def test_first_valid_pair_selection_supports_inside_out_directions(self) -> None:
        settings = make_settings()
        image = np.asarray(
            [
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
            ],
            dtype=np.float32,
        )

        scan = scan_raw_edge_candidates(image, (0, 0, 10, 1), "horizontal", settings)
        selected = select_first_valid_boundary_pairs_per_scanline(scan, "center_to_left", "center_to_right")

        self.assertEqual(len(selected), 2)
        self.assertEqual([pair.first.position for pair in selected], [3.5, 3.5])
        self.assertEqual([pair.second.position for pair in selected], [8.5, 8.5])

    def test_scan_edge_in_direction_supports_relative_direction_modes(self) -> None:
        candidates = [
            RawEdgeCandidate("horizontal", 0, 0, position, position, 0.0, strength, strength, 1, 0.0, strength)
            for position, strength in [(2.0, 40.0), (5.0, 90.0), (8.0, 60.0)]
        ]

        self.assertEqual(scan_edge_in_direction(candidates, "outside_to_center", from_left=True).position, 2.0)
        self.assertEqual(scan_edge_in_direction(candidates, "outside_to_center", from_left=False).position, 8.0)
        self.assertEqual(scan_edge_in_direction(candidates, "center_to_outside", from_left=True).position, 8.0)
        self.assertEqual(scan_edge_in_direction(candidates, "center_to_outside", from_left=False).position, 2.0)
        self.assertEqual(
            scan_edge_in_direction(candidates, "outside_to_center", from_left=True, prefer_strongest=True).position,
            5.0,
        )

    def test_nan_postprocessing_fills_small_gaps_and_smooths_finite_points(self) -> None:
        values = np.asarray([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0])

        filled = fill_small_gaps_nan(values, max_gap=2)
        smoothed = moving_average_nan(filled, win=3)

        np.testing.assert_allclose(filled[:4], [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(filled[4]))
        self.assertTrue(np.isnan(smoothed[4]))
        self.assertAlmostEqual(smoothed[1], 2.0, delta=1e-6)

    def test_taper_boundary_selection_uses_strongest_side_candidate_not_outermost_candidate(self) -> None:
        settings = make_settings()
        image = np.asarray(
            [
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
                [120, 85, 85, 85, 5, 5, 5, 45, 45, 140, 140],
            ],
            dtype=np.float32,
        )

        scan = scan_raw_edge_candidates(image, (0, 0, 10, 1), "horizontal", settings)
        left = select_strongest_boundary_candidates(scan, "left")
        right = select_strongest_boundary_candidates(scan, "right")

        self.assertEqual([candidate.position for candidate in left], [3.5, 3.5])
        self.assertEqual([candidate.position for candidate in right], [8.5, 8.5])

    def test_line_refinement_removes_isolated_selected_boundary_outlier(self) -> None:
        candidates = [
            RawEdgeCandidate(
                scan_axis="horizontal",
                scan_index=y,
                local_scan_index=y,
                position=float(10 + y if y != 5 else 50),
                image_x=float(10 + y if y != 5 else 50),
                image_y=float(y),
                strength=90.0,
                signed_delta=-90.0,
                sign=-1,
                grayscale_before=100.0,
                grayscale_after=10.0,
            )
            for y in range(10)
        ]

        refined = refine_boundary_candidates_by_line(candidates, residual_limit_px=2.0)

        self.assertEqual(len(refined), 9)
        self.assertNotIn(50.0, [candidate.position for candidate in refined])

    def test_vertical_refined_pair_selection_uses_same_side_split_as_horizontal(self) -> None:
        settings = MeasurementSettings(roi=(0, 0, 4, 9), minimum_grayscale_delta=20.0)
        image = np.full((10, 5), 90, dtype=np.float32)
        image[2:, :] = 30
        image[6:, 0:3] = 95
        image[8:, 3:5] = 140

        scan = scan_raw_edge_candidates(image, settings.roi, "vertical", settings)
        selected = select_refined_side_pair_per_scanline(scan)

        self.assertEqual([pair.scan_index for pair in selected], [0, 1, 2, 3, 4])
        self.assertEqual([pair.first.image_y for pair in selected], [1.5, 1.5, 1.5, 1.5, 1.5])
        np.testing.assert_allclose(
            [pair.second.image_y for pair in selected],
            [5.5, 5.5, 6.166666666666667, 6.833333333333333, 7.5],
            rtol=1e-6,
        )

    def test_limit_peak_taper_tie_breaks_by_scan_mode(self) -> None:
        settings = MeasurementSettings(minimum_grayscale_delta=20.0)
        engine = LimitPeakBoundaryEngine(settings)
        candidates = [
            RawEdgeCandidate("horizontal", 0, 0, position, position, 0.0, 100.0, 100.0, 1, 0.0, 100.0)
            for position in (2.0, 8.0)
        ]

        engine.scan_mode = "outside_to_center"
        self.assertEqual(engine._primary_candidate(candidates, "left").position, 2.0)
        self.assertEqual(engine._primary_candidate(candidates, "right").position, 8.0)

        engine.scan_mode = "center_to_outside"
        self.assertEqual(engine._primary_candidate(candidates, "left").position, 8.0)
        self.assertEqual(engine._primary_candidate(candidates, "right").position, 2.0)

    def test_double_taper_returns_successful_side_when_other_side_has_no_boundary(self) -> None:
        settings = MeasurementSettings(
            roi=(0, 0, 79, 59),
            measurement_type="taper_double",
            minimum_grayscale_delta=20.0,
            edge_scan_mode="center_to_outside",
        )
        image = np.full((60, 80), 40, dtype=np.uint8)
        for y in range(image.shape[0]):
            left_edge = 16 + y // 10
            image[y, :left_edge] = 180

        result = run_measurement(image, settings)

        self.assertIsNotNone(result.left_taper)
        self.assertIsNotNone(result.right_taper)
        self.assertNotEqual(result.left_taper.status, "Fail")
        self.assertEqual(result.right_taper.status, "Fail")
        self.assertNotEqual(result.status, "Fail")


if __name__ == "__main__":
    unittest.main()
