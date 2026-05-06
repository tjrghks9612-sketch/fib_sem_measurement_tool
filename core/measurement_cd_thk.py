from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.boundary_tracking import build_boundary_track, extract_edge_bands, interpolate_track
from fib_sem_measurement_tool.models.result import DistanceResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _sample_indices(length: int, count: int, margin_ratio: float = 0.12) -> np.ndarray:
    count = max(3, min(int(count), max(3, length)))
    start = int(round(length * margin_ratio))
    end = int(round(length * (1.0 - margin_ratio))) - 1
    if end <= start:
        start, end = 0, length - 1
    return np.unique(np.linspace(start, end, count).round().astype(int))


def _status(conf: float, th: float, fail: bool = False) -> str:
    if fail:
        return "Fail"
    if conf >= th:
        return "OK"
    if conf >= 60.0:
        return "Check"
    return "Review Needed"


def _filter(values: np.ndarray, strength: float) -> np.ndarray:
    if values.size < 4:
        return values
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 1e-6:
        return values
    lim = max(1.0, float(strength) * 1.4826 * mad)
    return values[np.abs(values - med) <= lim]


def _finalize(result: DistanceResult, values: List[float], pairs, left_track, right_track, settings, warnings):
    raw = np.asarray(values, dtype=np.float32)
    filt = _filter(raw, settings.advanced.outlier_rejection_strength)
    result.values_px = [float(v) for v in filt]
    result.boundary_pairs = pairs
    result.valid_count = int(filt.size)
    min_count = max(3, int(settings.advanced.minimum_valid_line_count))
    if result.valid_count < min_count:
        result.status = "Fail"
        result.warning_message = "; ".join(warnings + ["valid distance count below minimum"])
        return result
    result.mean_px = float(np.mean(filt)); result.max_px = float(np.max(filt)); result.min_px = float(np.min(filt)); result.median_px = float(np.median(filt)); result.std_px = float(np.std(filt))
    result.selected_method = settings.distance_method
    result.selected_px = result.max_px if settings.distance_method == "max" else result.min_px if settings.distance_method == "min" else result.mean_px
    coverage = min(left_track.coverage, right_track.coverage)
    std_penalty = 1.0 / (1.0 + result.std_px)
    variation = float(np.std(raw - np.median(raw))) if raw.size else 99.0
    outlier_ratio = 0.0 if raw.size == 0 else max(0.0, 1.0 - filt.size / raw.size)
    conf = 100.0 * (0.3 * coverage + 0.2 * min(left_track.smoothness, right_track.smoothness) + 0.15 * min(left_track.continuity, right_track.continuity) + 0.15 * std_penalty + 0.1 * (1.0 - min(1.0, outlier_ratio)) + 0.1 * (1.0 / (1.0 + variation)))
    result.confidence = float(max(0.0, min(100.0, conf)))
    result.status = _status(result.confidence, settings.advanced.confidence_threshold)
    if outlier_ratio > 0.45:
        warnings.append("too many outliers removed")
    result.warning_message = "; ".join(warnings)
    return result


def measure_horizontal_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = gray[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32, copy=False)
    ys = _sample_indices(crop.shape[0], settings.advanced.scan_line_count)
    result = DistanceResult(orientation="horizontal", selected_method=settings.distance_method, total_count=int(ys.size))
    scan_bands = []
    for i, ly in enumerate(ys):
        bands = extract_edge_bands(crop[ly, :], i, float(y1 + ly), float(x1), settings)
        scan_bands.append(bands)
    left = build_boundary_track(scan_bands, "left", settings.edge_reference, settings)
    right = build_boundary_track(scan_bands, "right", settings.edge_reference, settings)
    warnings = []
    if left is None:
        warnings.append("left boundary track not found")
    if right is None:
        warnings.append("right boundary track not found")
    if left is None or right is None:
        result.warning_message = "; ".join(warnings)
        return result
    grid = np.unique(np.concatenate([np.array([p[0] for p in left.points]), np.array([p[0] for p in right.points])]))
    lvals = interpolate_track(left, grid)
    rvals = interpolate_track(right, grid)
    valid = np.isfinite(lvals) & np.isfinite(rvals) & ((rvals - lvals) > 0.5)
    pairs = [(float(y1 + ys[int(g)]), float(x1 + lvals[i]), float(x1 + rvals[i])) for i, g in enumerate(grid[valid])]
    values = [p[2] - p[1] for p in pairs]
    return _finalize(result, values, pairs, left, right, settings, warnings)


def measure_vertical_thk(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = gray[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32, copy=False)
    xs = _sample_indices(crop.shape[1], settings.advanced.scan_line_count)
    result = DistanceResult(orientation="vertical", selected_method=settings.distance_method, total_count=int(xs.size))
    scan_bands = []
    for i, lx in enumerate(xs):
        bands = extract_edge_bands(crop[:, lx], i, float(x1 + lx), float(y1), settings)
        scan_bands.append(bands)
    top = build_boundary_track(scan_bands, "top", settings.edge_reference, settings)
    bottom = build_boundary_track(scan_bands, "bottom", settings.edge_reference, settings)
    warnings = []
    if top is None:
        warnings.append("top boundary track not found")
    if bottom is None:
        warnings.append("bottom boundary track not found")
    if top is None or bottom is None:
        result.warning_message = "; ".join(warnings)
        return result
    grid = np.unique(np.concatenate([np.array([p[0] for p in top.points]), np.array([p[0] for p in bottom.points])]))
    tvals = interpolate_track(top, grid)
    bvals = interpolate_track(bottom, grid)
    valid = np.isfinite(tvals) & np.isfinite(bvals) & ((bvals - tvals) > 0.5)
    pairs = [(float(x1 + xs[int(g)]), float(y1 + tvals[i]), float(y1 + bvals[i])) for i, g in enumerate(grid[valid])]
    values = [p[2] - p[1] for p in pairs]
    return _finalize(result, values, pairs, top, bottom, settings, warnings)
