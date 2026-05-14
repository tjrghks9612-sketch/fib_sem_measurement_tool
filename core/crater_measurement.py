from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fib_sem_measurement_tool.models.result import CraterResult, MeasurementResult, MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


Point = Tuple[float, float]


@dataclass
class _LineFit:
    line: Optional[Tuple[float, float, float, float]]
    angle_horizontal: Optional[float]
    angle_vertical: Optional[float]
    error: Optional[float]
    count: int
    status: str


def _empty_result(message: str) -> MeasurementResult:
    crater = CraterResult(status=MeasurementStatus.FAIL.value, warning_message=message)
    return MeasurementResult(
        measurement_type="crater",
        crater=crater,
        overall_confidence=0.0,
        status=MeasurementStatus.FAIL.value,
        warning_message=message,
    )


def _moving_median(values: np.ndarray, window: int = 9) -> np.ndarray:
    if values.size == 0:
        return values
    radius = max(1, int(window) // 2)
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(values.size):
        out[idx] = float(np.median(padded[idx : idx + radius * 2 + 1]))
    return out


def _baseline_candidates(grad_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grad_y.shape
    y1 = int(h * 0.34)
    y2 = int(h * 0.78)
    if y2 <= y1 + 4:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    columns = np.r_[np.arange(0, max(1, int(w * 0.24))), np.arange(min(w, int(w * 0.76)), w)]
    xs: List[float] = []
    ys: List[float] = []
    for x in columns:
        profile = grad_y[y1:y2, int(x)]
        if profile.size < 3:
            continue
        peak = int(np.argmax(profile)) + y1
        strength = float(grad_y[peak, int(x)])
        if strength <= 0:
            continue
        xs.append(float(x))
        ys.append(float(peak))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _fit_baseline(xs: np.ndarray, ys: np.ndarray, w: int) -> tuple[float, float, float, str]:
    if xs.size < max(8, int(w * 0.08)):
        return 0.0, 0.0, 0.0, MeasurementStatus.FAIL.value
    median_y = float(np.median(ys))
    mad = float(np.median(np.abs(ys - median_y))) + 1e-6
    inliers = np.abs(ys - median_y) <= max(8.0, mad * 2.5)
    if int(np.sum(inliers)) < max(8, int(xs.size * 0.35)):
        return 0.0, median_y, 35.0, MeasurementStatus.REVIEW_NEEDED.value
    slope, intercept = np.polyfit(xs[inliers], ys[inliers], 1)
    residual = float(np.median(np.abs((slope * xs[inliers] + intercept) - ys[inliers])))
    confidence = max(0.0, min(100.0, 100.0 - residual * 8.0))
    status = MeasurementStatus.OK.value if confidence >= 80.0 else MeasurementStatus.CHECK.value
    return float(slope), float(intercept), confidence, status


def _extract_top_profile(grad_y: np.ndarray, baseline_y: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grad_y.shape
    top_y = np.full(w, np.nan, dtype=np.float32)
    strengths = np.zeros(w, dtype=np.float32)
    valid = np.zeros(w, dtype=bool)
    global_floor = max(6.0, float(limit) * 0.45)
    for x in range(w):
        y_stop = int(min(h - 2, max(6, baseline_y[x] + h * 0.05)))
        y_start = int(max(1, h * 0.06))
        if y_stop <= y_start + 4:
            continue
        profile = grad_y[y_start:y_stop, x]
        if profile.size < 3:
            continue
        local_max = float(np.max(profile))
        threshold = max(global_floor, local_max * 0.42)
        peaks = []
        for local_idx in range(1, profile.size - 1):
            value = float(profile[local_idx])
            if value >= threshold and value >= float(profile[local_idx - 1]) and value >= float(profile[local_idx + 1]):
                peaks.append((local_idx + y_start, value))
        if not peaks:
            peak = int(np.argmax(profile)) + y_start
            value = float(grad_y[peak, x])
            if value < global_floor:
                continue
            peaks = [(peak, value)]
        strongest = max(peaks, key=lambda item: item[1])[1]
        comparable = [item for item in peaks if item[1] >= strongest * 0.74]
        # The upper comparable interface defines the dome, while flat regions stay on their stronger baseline edge.
        peak_y, strength = sorted(comparable, key=lambda item: item[0])[0]
        top_y[x] = float(peak_y)
        strengths[x] = float(strength)
        valid[x] = True
    if np.any(valid):
        filled = top_y.copy()
        indexes = np.arange(w, dtype=np.float32)
        filled[~valid] = np.interp(indexes[~valid], indexes[valid], top_y[valid])
        top_y = _moving_median(filled, 11)
    return top_y, strengths, valid


def _largest_height_region(heights: np.ndarray) -> Optional[tuple[int, int]]:
    finite = np.isfinite(heights)
    if not np.any(finite):
        return None
    max_height = float(np.nanmax(heights))
    if max_height < 6.0:
        return None
    threshold = max(5.0, max_height * 0.10)
    mask = finite & (heights >= threshold)
    runs: List[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    if not runs:
        return None
    center = int(np.nanargmax(heights))
    containing = [run for run in runs if run[0] <= center <= run[1]]
    return max(containing or runs, key=lambda run: run[1] - run[0])


def _profile_smoothness(profile_y: np.ndarray, left: int, right: int) -> float:
    segment = profile_y[left : right + 1]
    if segment.size < 4:
        return 0.0
    diffs = np.diff(segment.astype(np.float32))
    return float(np.median(np.abs(diffs)))


def _baseline_value(slope: float, intercept: float, x: np.ndarray | float) -> np.ndarray | float:
    return slope * x + intercept


def _fit_taper(points: Sequence[Point]) -> _LineFit:
    if len(points) < 6:
        return _LineFit(None, None, None, None, len(points), MeasurementStatus.FAIL.value)
    pts = np.asarray(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = np.abs((slope * xs + intercept) - ys)
    limit = max(3.0, float(np.median(residuals)) * 2.5)
    inliers = residuals <= limit
    if int(np.sum(inliers)) >= 6 and int(np.sum(inliers)) < len(points):
        slope, intercept = np.polyfit(xs[inliers], ys[inliers], 1)
        xs = xs[inliers]
        ys = ys[inliers]
    residual = float(np.sqrt(np.mean(((slope * xs + intercept) - ys) ** 2)))
    x1 = float(np.min(xs))
    x2 = float(np.max(xs))
    y1 = float(slope * x1 + intercept)
    y2 = float(slope * x2 + intercept)
    dx = x2 - x1
    dy = y2 - y1
    angle_h = abs(float(np.degrees(np.arctan2(dy, dx))))
    if angle_h > 90.0:
        angle_h = 180.0 - angle_h
    angle_v = abs(90.0 - angle_h)
    status = MeasurementStatus.OK.value if residual <= 8.0 else MeasurementStatus.CHECK.value
    return _LineFit((x1, y1, x2, y2), angle_h, angle_v, residual, int(xs.size), status)


def _status_from_confidence(confidence: float, fail: bool = False) -> str:
    if fail:
        return MeasurementStatus.FAIL.value
    if confidence >= 82.0:
        return MeasurementStatus.OK.value
    if confidence >= 62.0:
        return MeasurementStatus.CHECK.value
    return MeasurementStatus.REVIEW_NEEDED.value


def measure_crater(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = gray[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 60:
        return _empty_result("crater_roi_too_small")

    blurred = cv2.GaussianBlur(crop.astype(np.uint8), (5, 5), 0)
    grad_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    h, w = grad_y.shape
    limit = float(getattr(settings, "minimum_grayscale_delta", 55.0))

    base_xs, base_ys = _baseline_candidates(grad_y)
    slope, intercept, baseline_confidence, baseline_status = _fit_baseline(base_xs, base_ys, w)
    if baseline_status == MeasurementStatus.FAIL.value:
        return _empty_result("crater_baseline_not_found")

    xs = np.arange(w, dtype=np.float32)
    baseline_y = np.asarray(_baseline_value(slope, intercept, xs), dtype=np.float32)
    top_y, strengths, valid = _extract_top_profile(grad_y, baseline_y, limit)
    if not np.any(valid):
        return _empty_result("crater_top_profile_not_found")

    heights = baseline_y - top_y
    heights[~np.isfinite(heights)] = np.nan
    region = _largest_height_region(heights)
    if region is None:
        return _empty_result("crater_left_foot_not_found;crater_right_foot_not_found")
    left, right = region
    if right - left < max(20, int(w * 0.08)):
        return _empty_result("crater_profile_coverage_low")

    top_profile_points = [(float(x1 + x), float(y1 + top_y[x])) for x in range(left, right + 1) if np.isfinite(top_y[x])]
    coverage = float(len(top_profile_points) / max(1, right - left + 1) * 100.0)
    smoothness = _profile_smoothness(top_y, left, right)
    max_height = float(np.nanmax(heights[left : right + 1]))
    smooth_confidence = max(0.0, min(100.0, 100.0 - smoothness * 12.0))
    strength_values = strengths[left : right + 1][strengths[left : right + 1] > 0]
    strength_confidence = 0.0 if strength_values.size == 0 else max(0.0, min(100.0, float(np.median(strength_values)) / max(limit, 1.0) * 100.0))
    top_confidence = float(np.mean([coverage, smooth_confidence, strength_confidence]))
    top_status = _status_from_confidence(top_confidence)

    left_foot = (float(x1 + left), float(y1 + baseline_y[left]))
    right_foot = (float(x1 + right), float(y1 + baseline_y[right]))
    foot_confidence = min(100.0, max(0.0, (right - left) / max(1, w) * 160.0))
    foot_status = _status_from_confidence(foot_confidence)
    cd_px = float(np.hypot(right_foot[0] - left_foot[0], right_foot[1] - left_foot[1]))

    center_local = int(round((left + right) * 0.5))
    band = max(3, int((right - left + 1) * 0.04))
    band_left = max(left, center_local - band)
    band_right = min(right, center_local + band)
    height_band = heights[band_left : band_right + 1]
    height_band = height_band[np.isfinite(height_band)]
    if height_band.size == 0:
        return _empty_result("crater_thk_unstable")
    thk_median = float(np.median(height_band))
    thk_mean = float(np.mean(height_band))
    thk_max = float(np.max(height_band))
    thk_min = float(np.min(height_band))
    center_x = float(x1 + center_local)
    baseline_center = float(y1 + baseline_y[center_local])
    top_center = float(y1 + top_y[center_local])

    lower_ratio = 0.10
    upper_ratio = 0.80
    normalized_height = heights / max(max_height, 1.0)
    left_points = [
        (float(x1 + x), float(y1 + top_y[x]))
        for x in range(left, center_local + 1)
        if np.isfinite(top_y[x]) and lower_ratio <= normalized_height[x] <= upper_ratio
    ]
    right_points = [
        (float(x1 + x), float(y1 + top_y[x]))
        for x in range(center_local, right + 1)
        if np.isfinite(top_y[x]) and lower_ratio <= normalized_height[x] <= upper_ratio
    ]
    left_fit = _fit_taper(left_points)
    right_fit = _fit_taper(right_points)
    avg_taper = None
    taper_diff = None
    if left_fit.angle_horizontal is not None and right_fit.angle_horizontal is not None:
        avg_taper = float((left_fit.angle_horizontal + right_fit.angle_horizontal) * 0.5)
        taper_diff = float(abs(left_fit.angle_horizontal - right_fit.angle_horizontal))

    taper_scores = []
    for fit in (left_fit, right_fit):
        if fit.status == MeasurementStatus.FAIL.value:
            taper_scores.append(25.0)
        else:
            taper_scores.append(max(0.0, min(100.0, 100.0 - float(fit.error or 0.0) * 8.0)))
    overall = float(np.mean([baseline_confidence, top_confidence, foot_confidence, *taper_scores]))
    warnings = []
    if baseline_confidence < 65.0:
        warnings.append("crater_baseline_not_found")
    if coverage < 70.0:
        warnings.append("crater_profile_coverage_low")
    if left_fit.status != MeasurementStatus.OK.value:
        warnings.append("crater_left_taper_unstable")
    if right_fit.status != MeasurementStatus.OK.value:
        warnings.append("crater_right_taper_unstable")
    if smoothness > 5.0:
        warnings.append("crater_fit_error_high")
    status = _status_from_confidence(overall)
    if not warnings and status != MeasurementStatus.OK.value:
        warnings.append("crater_thk_unstable")

    px_to_real = float(settings.calibration.px_to_real or 1.0)
    crater = CraterResult(
        cd_px=cd_px,
        cd=cd_px * px_to_real,
        thk_px=thk_median,
        thk=thk_median * px_to_real,
        thk_mean_px=thk_mean,
        thk_max_px=thk_max,
        thk_min_px=thk_min,
        thk_median_px=thk_median,
        left_foot_x=left_foot[0],
        left_foot_y=left_foot[1],
        right_foot_x=right_foot[0],
        right_foot_y=right_foot[1],
        baseline_y_left=float(y1 + baseline_y[left]),
        baseline_y_right=float(y1 + baseline_y[right]),
        baseline_slope=slope,
        baseline_intercept=float(y1 + intercept - slope * x1),
        baseline_confidence=baseline_confidence,
        baseline_status=baseline_status,
        center_x=center_x,
        top_y_at_center=top_center,
        baseline_y_at_center=baseline_center,
        top_profile_point_count=w,
        top_profile_valid_count=len(top_profile_points),
        top_profile_coverage=coverage,
        top_profile_smoothness=smoothness,
        top_profile_confidence=top_confidence,
        top_profile_status=top_status,
        foot_confidence=foot_confidence,
        foot_status=foot_status,
        cd_status=MeasurementStatus.OK.value,
        cd_confidence=foot_confidence,
        thk_status=MeasurementStatus.OK.value if thk_median > 0 else MeasurementStatus.FAIL.value,
        thk_confidence=max(0.0, min(100.0, thk_median / max(1.0, max_height) * 100.0)),
        left_taper_angle_horizontal=left_fit.angle_horizontal,
        right_taper_angle_horizontal=right_fit.angle_horizontal,
        left_taper_angle_vertical=left_fit.angle_vertical,
        right_taper_angle_vertical=right_fit.angle_vertical,
        avg_taper_angle=avg_taper,
        taper_angle_diff=taper_diff,
        left_taper_fit_error=left_fit.error,
        right_taper_fit_error=right_fit.error,
        left_taper_valid_count=left_fit.count,
        right_taper_valid_count=right_fit.count,
        left_taper_status=left_fit.status,
        right_taper_status=right_fit.status,
        confidence=overall,
        overall_confidence=overall,
        status=status,
        warning_message=";".join(warnings),
        top_profile_points=top_profile_points,
        baseline_line=(float(x1 + left), float(y1 + baseline_y[left]), float(x1 + right), float(y1 + baseline_y[right])),
        cd_line=(left_foot[0], left_foot[1], right_foot[0], right_foot[1]),
        thk_line=(center_x, baseline_center, center_x, top_center),
        left_taper_line=left_fit.line,
        right_taper_line=right_fit.line,
    )
    return MeasurementResult(
        measurement_type="crater",
        crater=crater,
        avg_taper_angle=avg_taper,
        taper_angle_diff=taper_diff,
        overall_confidence=overall,
        status=status,
        warning_message=crater.warning_message,
    )
