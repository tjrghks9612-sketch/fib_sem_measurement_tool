from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fib_sem_measurement_tool.core.target_preprocessing import preprocess_target_roi
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


@dataclass
class _BaselineFit:
    slope: float
    intercept: float
    confidence: float
    status: str
    candidate_count: int
    coverage: float
    y_position_ratio: float
    y_range: float
    warning_code: str = ""


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


def _baseline_candidates(grad_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    h, w = grad_y.shape
    y1 = int(h * 0.28)
    y2 = int(h * 0.94)
    if y2 <= y1 + 4:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), 0, 0.0

    stages = (0.24, 0.34, 0.46, 0.62, 1.0)
    min_strength = max(1.0, float(np.percentile(grad_y[y1:y2, :], 72.0)))
    best: Optional[tuple[float, np.ndarray, np.ndarray, float]] = None
    total_candidate_count = 0

    for side_width in stages:
        if side_width >= 0.99:
            columns = np.arange(0, w)
        else:
            left_end = max(1, int(w * side_width))
            right_start = min(w, int(w * (1.0 - side_width)))
            columns = np.r_[np.arange(0, left_end), np.arange(right_start, w)]
        if columns.size < 3:
            continue

        candidates: list[tuple[float, float, float]] = []
        for x in columns:
            profile = grad_y[y1:y2, int(x)]
            if profile.size < 3:
                continue
            local_max = float(np.max(profile))
            threshold = max(min_strength, local_max * 0.35)
            for local_y in range(1, profile.size - 1):
                value = float(profile[local_y])
                if value < threshold:
                    continue
                if value >= float(profile[local_y - 1]) and value >= float(profile[local_y + 1]):
                    y = float(y1 + local_y)
                    # Keep weak preference for lower, stable floor-like edges without forbidding shallower ROIs.
                    candidates.append((float(x), y, value * (0.65 + 0.35 * (y / max(1, h - 1)))))
        total_candidate_count += len(candidates)
        if not candidates:
            continue

        ys_all = np.asarray([item[1] for item in candidates], dtype=np.float32)
        strengths = np.asarray([item[2] for item in candidates], dtype=np.float32)
        bin_size = max(4.0, h * 0.018)
        bins = np.round(ys_all / bin_size).astype(int)
        for bucket in np.unique(bins):
            mask = np.abs(bins - bucket) <= 1
            if int(np.sum(mask)) < max(6, int(columns.size * 0.04)):
                continue
            cluster_xs = np.asarray([candidates[i][0] for i in np.flatnonzero(mask)], dtype=np.float32)
            cluster_ys = ys_all[mask]
            cluster_strengths = strengths[mask]
            unique_columns = len(set(int(round(x)) for x in cluster_xs))
            coverage = float(unique_columns / max(1, columns.size))
            y_ratio = float(np.median(cluster_ys) / max(1, h - 1))
            left_cov = float(np.mean(cluster_xs <= w * 0.5))
            balance = 1.0 - abs(left_cov - 0.5) * 2.0
            y_range = float(np.percentile(cluster_ys, 90) - np.percentile(cluster_ys, 10))
            score = (
                coverage * 130.0
                + y_ratio * 70.0
                + max(0.0, balance) * 25.0
                + float(np.median(cluster_strengths)) * 0.2
                - y_range * 2.5
                - (0.0 if y_ratio >= 0.48 else 40.0)
            )
            if best is None or score > best[0]:
                best = (score, cluster_xs, cluster_ys, coverage)
        if best is not None and best[3] >= 0.18:
            break

    if best is None:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), total_candidate_count, 0.0
    _score, xs, ys, coverage = best
    return xs.astype(np.float32), ys.astype(np.float32), total_candidate_count, float(coverage)


def _fit_baseline(xs: np.ndarray, ys: np.ndarray, w: int, h: int, candidate_count: int, coverage: float) -> _BaselineFit:
    if xs.size < max(8, int(w * 0.08)):
        return _BaselineFit(0.0, 0.0, 0.0, MeasurementStatus.FAIL.value, int(candidate_count), float(coverage), 0.0, 0.0, "crater_baseline_coverage_low")
    median_y = float(np.median(ys))
    mad = float(np.median(np.abs(ys - median_y))) + 1e-6
    inliers = np.abs(ys - median_y) <= max(8.0, mad * 2.5)
    y_range = float(np.percentile(ys, 90) - np.percentile(ys, 10)) if ys.size else 0.0
    y_ratio = float(median_y / max(1, h - 1))
    if int(np.sum(inliers)) < max(8, int(xs.size * 0.35)):
        return _BaselineFit(0.0, median_y, 35.0, MeasurementStatus.REVIEW_NEEDED.value, int(candidate_count), float(coverage), y_ratio, y_range, "crater_baseline_unstable")
    slope, intercept = np.polyfit(xs[inliers], ys[inliers], 1)
    residual = float(np.median(np.abs((slope * xs[inliers] + intercept) - ys[inliers])))
    slope_penalty = abs(float(slope)) * 35.0
    coverage_penalty = max(0.0, 0.35 - float(coverage)) * 120.0
    high_penalty = max(0.0, 0.55 - y_ratio) * 150.0
    range_penalty = max(0.0, y_range - max(6.0, h * 0.04)) * 2.0
    confidence = max(0.0, min(100.0, 100.0 - residual * 8.0 - slope_penalty - coverage_penalty - high_penalty - range_penalty))
    warning = ""
    if coverage < 0.16:
        warning = "crater_baseline_coverage_low"
    elif y_ratio < 0.52:
        warning = "crater_baseline_may_be_too_high"
    elif y_range > max(10.0, h * 0.08):
        warning = "crater_baseline_unstable"
    status = _status_from_confidence(confidence)
    return _BaselineFit(float(slope), float(intercept), confidence, status, int(candidate_count), float(coverage), y_ratio, y_range, warning)


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


def _dark_object_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    source = values.astype(np.uint8, copy=False).reshape(-1, 1)
    otsu, _binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return min(float(otsu), float(np.percentile(values, 38.0)))


def _segments(mask_line: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(mask_line):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            result.append((start, idx - 1))
            start = None
    if start is not None:
        result.append((start, len(mask_line) - 1))
    return result


def _dark_profile_from_baseline(blurred: np.ndarray, baseline_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = blurred.shape
    samples = []
    top_limit = int(max(1, h * 0.06))
    for x in range(w):
        bottom = int(min(h - 1, max(top_limit + 2, baseline_y[x] - 2)))
        if bottom > top_limit:
            samples.append(blurred[top_limit:bottom, x])
    if not samples:
        return np.full(w, np.nan, dtype=np.float32), np.zeros(w, dtype=bool), np.zeros_like(blurred, dtype=bool)
    threshold = _dark_object_threshold(np.concatenate(samples))
    mask = blurred <= threshold
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)).astype(bool)
    top_y = np.full(w, np.nan, dtype=np.float32)
    valid = np.zeros(w, dtype=bool)
    for x in range(w):
        bottom = int(min(h - 1, max(top_limit + 2, baseline_y[x] - 3)))
        line = mask[top_limit:bottom, x]
        if line.size < 5:
            continue
        candidates = []
        for start, end in _segments(line):
            segment_height = end - start + 1
            distance_to_baseline = (line.size - 1) - end
            if segment_height < 4 or distance_to_baseline > max(12, int(h * 0.04)):
                continue
            candidates.append((segment_height - distance_to_baseline * 2.0, start, end))
        if not candidates:
            continue
        _score, start, _end = max(candidates, key=lambda item: item[0])
        top_y[x] = float(top_limit + start)
        valid[x] = True
    if np.sum(valid) >= 3:
        indexes = np.arange(w, dtype=np.float32)
        filled = top_y.copy()
        filled[~valid] = np.interp(indexes[~valid], indexes[valid], top_y[valid])
        top_y = _moving_median(filled, 13)
    return top_y, valid, mask


def _extract_side_boundaries(
    dark_mask: np.ndarray,
    top_y: np.ndarray,
    baseline_y: np.ndarray,
    left: int,
    right: int,
) -> tuple[list[Point], list[Point]]:
    h, w = dark_mask.shape
    if right <= left:
        return [], []
    finite_top = top_y[left : right + 1][np.isfinite(top_y[left : right + 1])]
    if finite_top.size == 0:
        return [], []
    y_start = max(0, int(np.floor(np.min(finite_top))))
    y_end = min(h - 1, int(np.ceil(np.median(baseline_y[left : right + 1]))))
    center = (left + right) * 0.5
    previous = (left, right)
    left_points: list[Point] = []
    right_points: list[Point] = []
    for y in range(y_start, y_end + 1):
        row = dark_mask[y, :]
        candidates = []
        for start, end in _segments(row):
            width = end - start + 1
            if width < max(6, int(w * 0.015)) or width > int(w * 0.90):
                continue
            overlap = max(0, min(end, previous[1]) - max(start, previous[0]) + 1)
            midpoint = (start + end) * 0.5
            if overlap <= 0 and abs(midpoint - center) > (right - left) * 0.62:
                continue
            score = width + overlap * 2.0 - abs(midpoint - center) * 0.2
            candidates.append((score, start, end))
        if not candidates:
            continue
        _score, start, end = max(candidates, key=lambda item: item[0])
        previous = (start, end)
        left_points.append((float(start), float(y)))
        right_points.append((float(end), float(y)))
    return left_points, right_points


def _largest_height_region(heights: np.ndarray) -> Optional[tuple[int, int]]:
    finite = np.isfinite(heights)
    if not np.any(finite):
        return None
    max_height = float(np.nanmax(heights))
    if max_height < 6.0:
        return None
    threshold = max(3.0, max_height * 0.0)
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

    processed = preprocess_target_roi(crop.astype(np.uint8))
    blurred = cv2.GaussianBlur(processed, (5, 5), 0)
    grad_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    h, w = grad_y.shape
    limit = float(getattr(settings, "minimum_grayscale_delta", 55.0))

    base_xs, base_ys, baseline_candidate_count, baseline_coverage = _baseline_candidates(grad_y)
    baseline_fit = _fit_baseline(base_xs, base_ys, w, h, baseline_candidate_count, baseline_coverage)
    slope = baseline_fit.slope
    intercept = baseline_fit.intercept
    baseline_confidence = baseline_fit.confidence
    baseline_status = baseline_fit.status
    if baseline_status == MeasurementStatus.FAIL.value:
        return _empty_result("crater_baseline_not_found")

    xs = np.arange(w, dtype=np.float32)
    baseline_y = np.asarray(_baseline_value(slope, intercept, xs), dtype=np.float32)
    scan_origin = getattr(settings, "crater_scan_origin", "auto")
    top_y, valid, dark_mask = _dark_profile_from_baseline(blurred, baseline_y)
    strengths = np.zeros(w, dtype=np.float32)
    for idx in np.flatnonzero(valid):
        yy = int(max(0, min(h - 1, round(float(top_y[idx])))))
        strengths[idx] = float(grad_y[yy, idx])
    if not np.any(valid) and scan_origin in {"auto", "from_top"}:
        top_y, strengths, valid = _extract_top_profile(grad_y, baseline_y, limit)
        dark_mask = blurred <= np.percentile(blurred, 40.0)
    if not np.any(valid):
        return _empty_result("crater_top_profile_not_found")

    heights = baseline_y - top_y
    heights[~np.isfinite(heights)] = np.nan
    region_heights = heights.copy()
    finite_region_heights = region_heights[np.isfinite(region_heights)]
    if finite_region_heights.size:
        low_height_cutoff = max(5.0, float(np.nanmax(finite_region_heights)) * 0.20)
        region_heights[(~valid) & (region_heights < low_height_cutoff)] = np.nan
    region = _largest_height_region(region_heights)
    if region is None:
        return _empty_result("crater_left_foot_not_found;crater_right_foot_not_found")
    left, right = region
    if right - left < max(20, int(w * 0.08)):
        return _empty_result("crater_profile_coverage_low")

    top_profile_points = [(float(x1 + x), float(y1 + top_y[x])) for x in range(left, right + 1) if np.isfinite(top_y[x])]
    left_boundary_local, right_boundary_local = _extract_side_boundaries(dark_mask, top_y, baseline_y, left, right)
    left_boundary_points = [(float(x1 + x), float(y1 + y)) for x, y in left_boundary_local]
    right_boundary_points = [(float(x1 + x), float(y1 + y)) for x, y in right_boundary_local]
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

    taper_height_percent = max(0.0, min(100.0, float(getattr(settings, "crater_taper_height_percent", 15.0))))
    lower_ratio = taper_height_percent / 100.0
    upper_ratio = 0.80 if lower_ratio < 0.80 else min(1.0, lower_ratio + 0.05)
    baseline_median = float(np.median(baseline_y[left : right + 1]))
    left_points = [
        (x, y)
        for x, y in left_boundary_points
        if lower_ratio <= (float(y1 + baseline_median) - y) / max(max_height, 1.0) <= upper_ratio
    ]
    right_points = [
        (x, y)
        for x, y in right_boundary_points
        if lower_ratio <= (float(y1 + baseline_median) - y) / max(max_height, 1.0) <= upper_ratio
    ]
    left_fit = _fit_taper(left_points)
    right_fit = _fit_taper(right_points)
    left_taper_measure_y = float(y1 + baseline_median - max_height * lower_ratio)
    right_taper_measure_y = left_taper_measure_y
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
    baseline_too_high = bool((baseline_fit.y_position_ratio or 0.0) < 0.54)
    dome_height_ratio = float(max_height / max(1, h))
    short_profile = bool((right - left + 1) < max(24, int(w * 0.18)))
    partial_dome = bool(
        baseline_too_high
        or dome_height_ratio < 0.16
        or (baseline_fit.coverage < 0.18 and dome_height_ratio < 0.28)
        or short_profile
    )

    overall = float(np.mean([baseline_confidence, top_confidence, foot_confidence, *taper_scores]))
    warnings = []
    if baseline_fit.warning_code:
        warnings.append(baseline_fit.warning_code)
    if baseline_confidence < 65.0:
        warnings.append("crater_roi_needs_more_baseline")
    if coverage < 70.0:
        warnings.append("crater_profile_coverage_low")
    if left_fit.status != MeasurementStatus.OK.value:
        warnings.append("crater_left_taper_unstable")
    if right_fit.status != MeasurementStatus.OK.value:
        warnings.append("crater_right_taper_unstable")
    if smoothness > 5.0:
        warnings.append("crater_fit_error_high")
    if partial_dome:
        warnings.append("crater_partial_dome_detected")
    status = _status_from_confidence(overall)
    if partial_dome and status == MeasurementStatus.OK.value:
        status = MeasurementStatus.CHECK.value
    if partial_dome and baseline_confidence < 60.0:
        status = MeasurementStatus.REVIEW_NEEDED.value
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
        baseline_candidate_count=baseline_fit.candidate_count,
        baseline_coverage=baseline_fit.coverage,
        baseline_confidence=baseline_confidence,
        baseline_status=baseline_status,
        baseline_y_position_ratio=baseline_fit.y_position_ratio,
        baseline_y_range=baseline_fit.y_range,
        baseline_warning_code=baseline_fit.warning_code,
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
        taper_height_percent=taper_height_percent,
        left_taper_measure_y=left_taper_measure_y,
        right_taper_measure_y=right_taper_measure_y,
        left_taper_status=left_fit.status,
        right_taper_status=right_fit.status,
        confidence=overall,
        overall_confidence=overall,
        status=status,
        warning_message=";".join(dict.fromkeys(warnings)),
        top_profile_points=top_profile_points,
        left_boundary_points=left_boundary_points,
        right_boundary_points=right_boundary_points,
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
