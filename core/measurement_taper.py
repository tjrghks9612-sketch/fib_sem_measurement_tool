from __future__ import annotations

from typing import Sequence

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    refine_boundary_candidates_by_line,
    scan_raw_edge_candidates,
    select_first_valid_boundary_candidates,
)
from fib_sem_measurement_tool.models.result import MeasurementResult, RawEdgeCandidate, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


MIN_TAPER_POINTS = 2
MIN_COVERAGE_FOR_OK = 0.45


def _status_from_points(point_count: int, coverage: float) -> str:
    if point_count < MIN_TAPER_POINTS:
        return "Fail"
    if coverage >= MIN_COVERAGE_FOR_OK:
        return "OK"
    return "Check"


def _result_with_scan_metadata(side: str, scan) -> TaperSideResult:
    return TaperSideResult(
        side=side,
        raw_edge_candidates=list(scan.raw_edge_candidates),
        raw_edge_count=scan.raw_edge_count,
        scanned_line_count=scan.scanned_line_count,
        valid_scanline_count=scan.valid_scanline_count,
        scanline_coverage=scan.scanline_coverage,
        raw_edge_density=scan.raw_edge_density,
        per_scanline_candidate_count=dict(scan.per_scanline_candidate_count),
        minimum_grayscale_delta=scan.minimum_grayscale_delta,
    )


def _fit_selected_boundary(result: TaperSideResult, selected: Sequence[RawEdgeCandidate]) -> TaperSideResult:
    result.selected_boundary_candidates = list(selected)
    result.points = [(float(candidate.image_x), float(candidate.image_y)) for candidate in selected]
    result.selected_point_count = len(result.points)
    result.fit_point_count = len(result.points)
    result.valid_point_count = len(result.points)
    result.inlier_count = len(result.points)
    result.confidence = float(result.scanline_coverage * 100.0)
    result.status = _status_from_points(result.valid_point_count, result.scanline_coverage)

    if len(result.points) < MIN_TAPER_POINTS:
        result.warning_message = f"{result.side} raw grayscale edge points not found"
        return result

    pts = np.asarray(result.points, dtype=np.float64)
    xs, ys = pts[:, 0], pts[:, 1]
    m, b = np.polyfit(ys, xs, 1)
    pred = m * ys + b
    residuals = xs - pred
    err = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((xs - np.mean(xs)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-9 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    angle_horizontal = abs(float(np.degrees(np.arctan2(1.0, m))))
    if angle_horizontal > 90.0:
        angle_horizontal = 180.0 - angle_horizontal

    result.angle_horizontal = angle_horizontal
    result.angle_vertical = abs(90.0 - angle_horizontal)
    result.fit_r2 = float(r2)
    result.fit_error = err
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    result.fit_line = (float(m * y_min + b), y_min, float(m * y_max + b), y_max)
    return result


def _taper_edge_direction(side: str, settings: MeasurementSettings) -> str:
    if side == "right":
        return getattr(settings, "cd_right_edge_direction", "right_to_center")
    return getattr(settings, "cd_left_edge_direction", "left_to_center")


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    scan = scan_raw_edge_candidates(gray, roi, "horizontal", settings)
    result = _result_with_scan_metadata(side, scan)
    selected = refine_boundary_candidates_by_line(
        select_first_valid_boundary_candidates(scan, side, _taper_edge_direction(side, settings))
    )
    return _fit_selected_boundary(result, selected)


def measure_single_taper(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> MeasurementResult:
    taper = measure_taper_side(gray, roi, side, settings)
    result = MeasurementResult(
        measurement_type="taper_single",
        overall_confidence=taper.confidence,
        status=taper.status,
        warning_message=taper.warning_message,
    )
    if side == "right":
        result.right_taper = taper
    else:
        result.left_taper = taper
    return result


def measure_double_taper(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    left = measure_taper_side(gray, roi, "left", settings)
    right = measure_taper_side(gray, roi, "right", settings)
    valid = [taper for taper in (left, right) if taper.status != "Fail"]
    overall = float(np.mean([taper.confidence for taper in valid])) if valid else 0.0
    angles = [taper.angle_horizontal for taper in valid if taper.angle_horizontal is not None]
    avg = float(np.mean(angles)) if angles else None
    diff = (
        abs(left.angle_horizontal - right.angle_horizontal)
        if left.angle_horizontal is not None and right.angle_horizontal is not None
        else None
    )
    status = "Fail" if not valid else "OK" if overall >= MIN_COVERAGE_FOR_OK * 100.0 else "Check"
    return MeasurementResult(
        measurement_type="taper_double",
        left_taper=left,
        right_taper=right,
        avg_taper_angle=avg,
        taper_angle_diff=diff,
        overall_confidence=overall,
        status=status,
        warning_message="; ".join([msg for msg in [left.warning_message, right.warning_message] if msg]),
    )
