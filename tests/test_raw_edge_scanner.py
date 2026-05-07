import unittest
from unittest.mock import patch

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    detect_profile_candidates,
    refine_boundary_candidates_by_line,
    scan_pair_candidates,
    scan_raw_edge_candidates,
    select_first_valid_boundary_pairs_per_scanline,
    select_refined_side_pair_per_scanline,
    select_strongest_boundary_candidates,
)
from fib_sem_measurement_tool.models.result import RawEdgeCandidate
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
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

    def test_threshold_includes_only_jumps_at_or_above_minimum_delta(self) -> None:
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

        self.assertEqual([candidate.position for candidate in candidates], [1.5, 3.5])
        self.assertEqual([candidate.signed_delta for candidate in candidates], [25.0, 26.0])

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

    def test_refined_pair_selection_uses_first_valid_candidate_in_each_half(self) -> None:
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
        self.assertEqual([pair.first.position for pair in selected], [0.5, 0.5])
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
        self.assertEqual([pair.second.position for pair in selected], [6.5, 6.5])

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
        self.assertEqual([pair.second.image_y for pair in selected], [5.5, 5.5, 5.5, 7.5, 7.5])


if __name__ == "__main__":
    unittest.main()
