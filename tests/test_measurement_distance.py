from __future__ import annotations

import pytest
import numpy as np

from fib_sem_measurement_tool.core.boundary_tracking_selftest import run_core_selftest
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.models.result import MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _settings(measurement_type: str, roi: tuple[int, int, int, int]) -> MeasurementSettings:
    settings = MeasurementSettings(roi=roi, measurement_type=measurement_type, noise_level="low")
    settings.apply_noise_preset(force=True)
    settings.edge_reference = "inner"
    settings.distance_method = "mean"
    settings.advanced.scan_line_count = 9
    settings.advanced.minimum_valid_line_count = 4
    settings.advanced.min_valid_line_ratio = 0.35
    return settings


def test_horizontal_distance_succeeds_for_offset_roi() -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[20:100, 60:104] = 220
    settings = _settings("distance_horizontal", (10, 10, 170, 110))

    result = run_measurement(image, settings)

    assert result.horizontal_cd is not None
    assert result.status != MeasurementStatus.FAIL.value, result.warning_message
    assert result.horizontal_cd.valid_count >= settings.advanced.minimum_valid_line_count
    assert result.horizontal_cd.selected_px == pytest.approx(44.0, abs=8.0)


def test_vertical_distance_succeeds_for_offset_roi() -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[38:84, 30:150] = 220
    settings = _settings("distance_vertical", (10, 10, 170, 110))

    result = run_measurement(image, settings)

    assert result.vertical_thk is not None
    assert result.status != MeasurementStatus.FAIL.value, result.warning_message
    assert result.vertical_thk.valid_count >= settings.advanced.minimum_valid_line_count
    assert result.vertical_thk.selected_px == pytest.approx(46.0, abs=8.0)


def test_core_selftest_reports_distance_stats_and_edge_reference_variation() -> None:
    np.random.seed(0)

    report = run_core_selftest()

    assert report["horizontal_has_stats"] is True
    assert report["vertical_has_stats"] is True
    assert report["edge_reference_diff"] is True
