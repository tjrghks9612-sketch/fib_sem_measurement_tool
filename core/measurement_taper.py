from __future__ import annotations

from typing import Sequence

import numpy as np

from fib_sem_measurement_tool.core.boundary_tracking import build_boundary_track, extract_edge_bands
from fib_sem_measurement_tool.core.measurement_cd_thk import _sample_indices
from fib_sem_measurement_tool.models.result import MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _status(conf: float, th: float, fail: bool = False) -> str:
    if fail:
        return "Fail"
    if conf >= th:
        return "OK"
    if conf >= 60.0:
        return "Check"
    return "Review Needed"


def _fit_track(track, side: str, settings: MeasurementSettings) -> TaperSideResult:
    res = TaperSideResult(side=side)
    if track is None or len(track.points) < 3:
        res.warning_message = f"{side} boundary track not found"
        return res
    pts = np.asarray(track.points, dtype=np.float64)
    ys, xs = pts[:, 0], pts[:, 1]
    med = np.median(xs)
    mad = np.median(np.abs(xs - med)) + 1e-6
    lim = settings.advanced.outlier_rejection_strength * 1.4826 * mad
    keep = np.abs(xs - med) <= max(2.0, lim)
    ys, xs = ys[keep], xs[keep]
    if xs.size < 3:
        res.warning_message = "taper fit error too high"
        return res
    m, b = np.polyfit(ys, xs, 1)
    pred = m * ys + b
    err = float(np.sqrt(np.mean((xs - pred) ** 2)))
    ang_h = abs(float(np.degrees(np.arctan2(1.0, m))))
    if ang_h > 90:
        ang_h = 180 - ang_h
    res.angle_horizontal = ang_h
    res.angle_vertical = abs(90.0 - ang_h)
    res.fit_error = err
    res.valid_point_count = int(xs.size)
    res.inlier_count = int(xs.size)
    res.points = [(float(x), float(y)) for y, x in zip(ys, xs)]
    res.fit_line = (float(m * np.min(ys) + b), float(np.min(ys)), float(m * np.max(ys) + b), float(np.max(ys)))
    conf = 100.0 * (0.35 * track.coverage + 0.2 * track.smoothness + 0.15 * track.continuity + 0.15 * (1.0 / (1.0 + err)) + 0.15 * min(1.0, track.mean_strength / (track.mean_strength + 1.0)))
    res.confidence = float(max(0.0, min(100.0, conf)))
    res.status = _status(res.confidence, settings.advanced.confidence_threshold)
    return res


def _collect_track(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings):
    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = gray[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32, copy=False)
    ys = _sample_indices(crop.shape[0], settings.advanced.scan_line_count)
    scan_bands = [extract_edge_bands(crop[ly, :], i, float(y1 + ly), float(x1), settings) for i, ly in enumerate(ys)]
    return build_boundary_track(scan_bands, side, settings.edge_reference, settings)


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    return _fit_track(_collect_track(gray, roi, side, settings), side, settings)


def measure_single_taper(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> MeasurementResult:
    taper = measure_taper_side(gray, roi, side, settings)
    result = MeasurementResult(measurement_type="taper_single", overall_confidence=taper.confidence, status=_status(taper.confidence, settings.advanced.confidence_threshold, fail=taper.status == "Fail"), warning_message=taper.warning_message)
    if side == "right":
        result.right_taper = taper
    else:
        result.left_taper = taper
    return result


def measure_double_taper(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    left = measure_taper_side(gray, roi, "left", settings)
    right = measure_taper_side(gray, roi, "right", settings)
    valid = [t for t in (left, right) if t.status != "Fail"]
    overall = float(np.mean([t.confidence for t in valid])) if valid else 0.0
    avg = float(np.mean([t.angle_horizontal for t in valid if t.angle_horizontal is not None])) if valid else None
    diff = abs(left.angle_horizontal - right.angle_horizontal) if left.angle_horizontal is not None and right.angle_horizontal is not None else None
    return MeasurementResult(measurement_type="taper_double", left_taper=left, right_taper=right, avg_taper_angle=avg, taper_angle_diff=diff, overall_confidence=overall, status=_status(overall, settings.advanced.confidence_threshold, fail=not valid), warning_message="; ".join([w for w in [left.warning_message, right.warning_message] if w]))
