from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.confidence import fit_confidence
from fib_sem_measurement_tool.core.edge_detection import detect_edge_candidates, group_edge_bands, select_edge_from_band
from fib_sem_measurement_tool.core.measurement_cd_thk import _sample_indices
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _status_by_threshold(confidence: float, threshold: float, failed: bool = False) -> str:
    if failed:
        return "Fail"
    if confidence >= threshold:
        return "OK"
    if confidence >= max(0.0, threshold - 20.0):
        return "Check"
    return "Review Needed"


def _fit_line(points: List[Tuple[float, float]], side: str, settings: MeasurementSettings) -> TaperSideResult:
    result = TaperSideResult(side=side)
    if len(points) < 3:
        result.warning_message = "유효 taper point 부족"; return result
    pts = np.asarray(points, dtype=np.float64)
    xs, ys = pts[:, 0], pts[:, 1]
    slope, intercept = np.polyfit(ys, xs, 1)
    residuals = xs - (slope * ys + intercept)
    fit_error = float(np.sqrt(np.mean(residuals**2)))
    ss_tot = float(np.sum((xs - np.mean(xs)) ** 2)); ss_res = float(np.sum(residuals**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    result.fit_r2 = float(max(0.0, min(1.0, r2))); result.fit_error = fit_error
    result.valid_point_count = len(points); result.inlier_count = len(points)
    result.points = [(float(x), float(y)) for x, y in pts]
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    result.fit_line = (float(slope * y_min + intercept), y_min, float(slope * y_max + intercept), y_max)
    angle_h = abs(float(np.degrees(np.arctan2(1.0, slope))))
    result.angle_horizontal = angle_h if angle_h <= 90 else 180 - angle_h
    result.angle_vertical = abs(90.0 - result.angle_horizontal)
    result.confidence = fit_confidence(result.inlier_count, len(points), fit_error, result.fit_r2, max(3, int(settings.advanced.minimum_valid_line_count * 0.6)), max(3.0, settings.advanced.fit_error_threshold))
    result.status = _status_by_threshold(result.confidence, settings.advanced.confidence_threshold)
    return result


def _pick_side_band(profile: np.ndarray, side: str, settings: MeasurementSettings):
    bands = group_edge_bands(detect_edge_candidates(profile, settings), settings)
    if not bands:
        return None
    center = len(profile) / 2.0
    if side == "left":
        candidates = [b for b in bands if b["center"] <= center] or bands
        band = min(candidates, key=lambda b: b["center"]) if settings.edge_reference == "outer" else max(candidates, key=lambda b: b["center"]) if settings.edge_reference == "inner" else max(candidates, key=lambda b: b["strength"])
    else:
        candidates = [b for b in bands if b["center"] >= center] or bands
        band = max(candidates, key=lambda b: b["center"]) if settings.edge_reference == "outer" else min(candidates, key=lambda b: b["center"]) if settings.edge_reference == "inner" else max(candidates, key=lambda b: b["strength"])
    return select_edge_from_band(band, settings.edge_reference, side)


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
    result = TaperSideResult(side=side)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"; return result
    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    y_indices = _sample_indices(crop.shape[0], settings.advanced.scan_line_count, margin_ratio=0.10)
    points: List[Tuple[float, float]] = []
    for ly in y_indices:
        edge = _pick_side_band(crop[ly, :], side, settings)
        if edge is None:
            continue
        points.append((float(x1 + edge), float(y1 + ly)))
    return _fit_line(points, side, settings)


def measure_single_taper(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> MeasurementResult:
    taper = measure_taper_side(gray, roi, side, settings)
    result = MeasurementResult(measurement_type="taper_single", overall_confidence=taper.confidence, status=_status_by_threshold(taper.confidence, settings.advanced.confidence_threshold, failed=taper.status == "Fail"))
    result.warning_message = taper.warning_message
    if side == "right": result.right_taper = taper
    else: result.left_taper = taper
    return result


def measure_double_taper(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    left = measure_taper_side(gray, roi, "left", settings); right = measure_taper_side(gray, roi, "right", settings)
    valid = [item for item in (left, right) if item.status != "Fail"]
    overall = float(np.mean([item.confidence for item in valid])) if valid else 0.0
    result = MeasurementResult(measurement_type="taper_double", left_taper=left, right_taper=right, overall_confidence=overall, status=_status_by_threshold(overall, settings.advanced.confidence_threshold, failed=not valid), warning_message="; ".join([i.warning_message for i in (left, right) if i.warning_message]))
    return result
