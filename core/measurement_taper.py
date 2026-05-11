from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    refine_boundary_candidates_by_line,
    scan_raw_edge_candidates,
    select_boundary_curve_candidates,
)
from fib_sem_measurement_tool.models.result import MeasurementResult, RawEdgeCandidate, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


MIN_TAPER_POINTS = 2
MIN_COVERAGE_FOR_OK = 0.45
TAPER_TARGET_WINDOW_PCT = 18.0
TAPER_MIN_LOCAL_POINTS = 5
TAPER_RESIDUAL_LIMIT_PX = 18.0
TAPER_FIT_PREFILTER_RESIDUAL_PX = 6.0


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


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _target_y_for_side(roi: Sequence[int], side: str, settings: MeasurementSettings) -> float:
    _x1, y1, _x2, y2 = [int(v) for v in roi]
    base_pct = float(getattr(settings, "base_height_pct", 50.0))
    offset_pct = float(
        getattr(settings, "right_offset_pct", 0.0)
        if side == "right"
        else getattr(settings, "left_offset_pct", 0.0)
    )
    target_pct = _clamp_pct(base_pct + offset_pct)
    return float(y2) - (float(y2 - y1) * target_pct / 100.0)


def _fit_points_near_target(points: np.ndarray, target_y: float, roi: Sequence[int]) -> np.ndarray:
    if points.shape[0] <= MIN_TAPER_POINTS:
        return points

    _x1, y1, _x2, y2 = [int(v) for v in roi]
    height = max(1.0, float(y2 - y1 + 1))
    radius = max(4.0, height * TAPER_TARGET_WINDOW_PCT / 200.0)
    distances = np.abs(points[:, 1] - float(target_y))
    local = points[distances <= radius]
    if local.shape[0] >= MIN_TAPER_POINTS:
        return local

    needed = min(max(TAPER_MIN_LOCAL_POINTS, MIN_TAPER_POINTS), points.shape[0])
    nearest_indices = np.argsort(distances)[:needed]
    return points[np.sort(nearest_indices)]


def _fit_candidates_near_target(
    candidates: Sequence[RawEdgeCandidate],
    target_y: float,
    roi: Sequence[int],
) -> list[RawEdgeCandidate]:
    if len(candidates) <= MIN_TAPER_POINTS:
        return list(candidates)

    points = np.asarray([(candidate.image_x, candidate.image_y) for candidate in candidates], dtype=np.float64)
    _x1, y1, _x2, y2 = [int(v) for v in roi]
    height = max(1.0, float(y2 - y1 + 1))
    radius = max(4.0, height * TAPER_TARGET_WINDOW_PCT / 200.0)
    distances = np.abs(points[:, 1] - float(target_y))
    indices = np.flatnonzero(distances <= radius)
    if indices.size < MIN_TAPER_POINTS:
        needed = min(max(TAPER_MIN_LOCAL_POINTS, MIN_TAPER_POINTS), len(candidates))
        indices = np.argsort(distances)[:needed]
    return [candidates[int(index)] for index in np.sort(indices)]


def _prefilter_taper_fit_candidates(candidates: Sequence[RawEdgeCandidate]) -> list[RawEdgeCandidate]:
    selected = list(candidates)
    if len(selected) < TAPER_MIN_LOCAL_POINTS:
        return selected

    refined = refine_boundary_candidates_by_line(
        selected,
        residual_limit_px=TAPER_FIT_PREFILTER_RESIDUAL_PX,
        mad_multiplier=2.5,
        iterations=5,
    )
    if len(refined) < TAPER_MIN_LOCAL_POINTS:
        return selected
    if len(refined) < max(TAPER_MIN_LOCAL_POINTS, int(len(selected) * 0.35)):
        return selected
    return refined


def _line_angles(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    angle_horizontal = abs(math.degrees(math.atan2(dy, dx)))
    if angle_horizontal > 90.0:
        angle_horizontal = 180.0 - angle_horizontal
    angle_vertical = abs(math.degrees(math.atan2(dx, dy)))
    if angle_vertical > 90.0:
        angle_vertical = 180.0 - angle_vertical
    return float(angle_horizontal), float(angle_vertical)


def _fit_selected_boundary(
    result: TaperSideResult,
    selected: Sequence[RawEdgeCandidate],
    roi: Sequence[int],
    settings: MeasurementSettings,
) -> TaperSideResult:
    all_selected = list(selected)
    result.selected_boundary_candidates = list(all_selected)
    result.points = [(float(candidate.image_x), float(candidate.image_y)) for candidate in all_selected]
    result.selected_point_count = len(all_selected)

    if len(all_selected) < MIN_TAPER_POINTS:
        result.fit_point_count = len(all_selected)
        result.valid_point_count = len(all_selected)
        result.inlier_count = len(all_selected)
        result.confidence = 0.0
        result.status = _status_from_points(result.valid_point_count, 0.0)
        result.warning_message = f"{result.side} raw grayscale edge points not found"
        return result

    fit_source_candidates = _prefilter_taper_fit_candidates(all_selected)
    fit_candidates = list(fit_source_candidates)
    fit_pts = np.asarray([(candidate.image_x, candidate.image_y) for candidate in fit_candidates], dtype=np.float64)
    result.fit_point_count = int(fit_pts.shape[0])
    result.valid_point_count = int(fit_pts.shape[0])
    selected_coverage = float(len(all_selected) / result.scanned_line_count) if result.scanned_line_count else 0.0
    result.confidence = float(selected_coverage * 100.0)
    result.status = _status_from_points(result.valid_point_count, selected_coverage)

    if fit_pts.shape[0] < MIN_TAPER_POINTS:
        result.warning_message = f"{result.side} target taper fit points not found"
        return result

    xs, ys = fit_pts[:, 0], fit_pts[:, 1]
    m, b = np.polyfit(ys, xs, 1)
    pred = m * ys + b
    residuals = xs - pred
    err = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((xs - np.mean(xs)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-9 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    result.fit_r2 = float(r2)
    result.fit_error = err

    residual_limit = max(4.0, float(getattr(settings, "taper_residual_limit_px", TAPER_RESIDUAL_LIMIT_PX)))
    all_pts = np.asarray([(candidate.image_x, candidate.image_y) for candidate in all_selected], dtype=np.float64)
    all_residuals = np.abs(all_pts[:, 0] - (m * all_pts[:, 1] + b))
    inliers = [
        candidate
        for candidate, residual in zip(all_selected, all_residuals)
        if float(residual) <= residual_limit
    ]
    if len(inliers) < MIN_TAPER_POINTS:
        inliers = list(fit_candidates)
    result.selected_boundary_candidates = list(inliers)
    result.points = [(float(candidate.image_x), float(candidate.image_y)) for candidate in inliers]
    result.selected_point_count = len(result.points)
    result.inlier_count = len(result.points)

    line_pts = np.asarray(result.points, dtype=np.float64) if result.points else fit_pts
    y_min = float(np.min(line_pts[:, 1]))
    y_max = float(np.max(line_pts[:, 1]))
    x_min = float(m * y_min + b)
    x_max = float(m * y_max + b)
    result.angle_horizontal, result.angle_vertical = _line_angles(x_min, y_min, x_max, y_max)
    result.fit_line = (x_min, y_min, x_max, y_max)
    return result


def _taper_edge_direction(side: str, settings: MeasurementSettings) -> str:
    edge_scan_mode = getattr(settings, "edge_scan_mode", "auto")
    if edge_scan_mode == "outside_to_center":
        return "right_to_center" if side == "right" else "left_to_center"
    if edge_scan_mode == "center_to_outside":
        return "center_to_right" if side == "right" else "center_to_left"
    if side == "right":
        return getattr(settings, "taper_right_edge_direction", "center_to_right")
    return getattr(settings, "taper_left_edge_direction", "center_to_left")


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    scan = scan_raw_edge_candidates(gray, roi, "horizontal", settings)
    result = _result_with_scan_metadata(side, scan)
    selected = select_boundary_curve_candidates(
        scan,
        side,
        _taper_edge_direction(side, settings),
        max_jump_px=float(getattr(settings, "max_jump_px", 28.0)),
        prefer_sign=True,
    )
    return _fit_selected_boundary(result, selected, roi, settings)


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
