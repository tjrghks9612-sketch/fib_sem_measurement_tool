from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.confidence import distance_confidence
from fib_sem_measurement_tool.core.edge_detection import detect_edge_candidates, group_edge_bands, select_edge_from_band
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import DistanceResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _sample_indices(length: int, count: int, margin_ratio: float = 0.12) -> np.ndarray:
    count = max(3, min(int(count), max(3, length)))
    start = int(round(length * margin_ratio))
    end = int(round(length * (1.0 - margin_ratio))) - 1
    if end <= start:
        start, end = 0, length - 1
    return np.unique(np.linspace(start, end, count).round().astype(int))


def _status_by_threshold(confidence: float, threshold: float, failed: bool = False) -> str:
    if failed:
        return "Fail"
    if confidence >= threshold:
        return "OK"
    if confidence >= max(0.0, threshold - 20.0):
        return "Check"
    return "Review Needed"


def _filter_distances(values: List[float], pairs: List[Tuple[float, float, float]], strength: float) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    array = np.asarray(values, dtype=np.float32)
    if array.size < 4:
        return array, pairs
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= 1e-6:
        return array, pairs
    limit = max(2.0, float(strength) * 1.4826 * mad)
    keep = np.abs(array - median) <= limit
    return array[keep], [pair for pair, ok in zip(pairs, keep) if bool(ok)]


def _extract_pair(profile: np.ndarray, settings: MeasurementSettings, primary: str, secondary: str) -> Tuple[float, float] | None:
    bands = group_edge_bands(detect_edge_candidates(profile, settings), settings)
    if len(bands) < 2:
        return None
    sorted_bands = sorted(bands, key=lambda b: b["center"])
    if primary in {"left", "top"}:
        first, second = sorted_bands[0], sorted_bands[-1]
    else:
        first, second = sorted_bands[-1], sorted_bands[0]
    a = select_edge_from_band(first, settings.edge_reference, primary)
    b = select_edge_from_band(second, settings.edge_reference, secondary)
    left, right = (a, b) if a <= b else (b, a)
    if right - left <= 2:
        return None
    return float(left), float(right)


def _finish(result: DistanceResult, values: List[float], pairs: List[Tuple[float, float, float]], settings: MeasurementSettings) -> DistanceResult:
    filtered, filtered_pairs = _filter_distances(values, pairs, settings.advanced.outlier_rejection_strength)
    result.values_px = [float(v) for v in filtered]
    result.boundary_pairs = filtered_pairs
    result.valid_count = int(filtered.size)
    min_count = max(3, min(int(settings.advanced.minimum_valid_line_count), max(3, result.total_count)))
    if result.valid_count < min_count:
        result.confidence = 0.0
        result.status = "Fail"
        return result
    result.mean_px = float(np.mean(filtered)); result.max_px = float(np.max(filtered)); result.min_px = float(np.min(filtered)); result.median_px = float(np.median(filtered)); result.std_px = float(np.std(filtered))
    result.selected_px = result.max_px if settings.distance_method == "max" else result.min_px if settings.distance_method == "min" else result.mean_px
    result.selected_method = settings.distance_method
    result.confidence = distance_confidence(filtered, result.valid_count, result.total_count, min_count, float(settings.advanced.min_valid_line_ratio))
    result.status = _status_by_threshold(result.confidence, settings.advanced.confidence_threshold)
    return result


def measure_horizontal_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
    result = DistanceResult(orientation="horizontal", selected_method=settings.distance_method)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"; return result
    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    y_indices = _sample_indices(crop.shape[0], settings.advanced.scan_line_count)
    result.total_count = int(y_indices.size)
    values: List[float] = []; pairs: List[Tuple[float, float, float]] = []
    for ly in y_indices:
        pair = _extract_pair(crop[ly, :], settings, "left", "right")
        if pair is None:
            continue
        left_x, right_x = float(x1 + pair[0]), float(x1 + pair[1])
        values.append(right_x - left_x)
        pairs.append((float(y1 + ly), left_x, right_x))
    return _finish(result, values, pairs, settings)


def measure_vertical_thk(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
    result = DistanceResult(orientation="vertical", selected_method=settings.distance_method)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"; return result
    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    x_indices = _sample_indices(crop.shape[1], settings.advanced.scan_line_count)
    result.total_count = int(x_indices.size)
    values: List[float] = []; pairs: List[Tuple[float, float, float]] = []
    for lx in x_indices:
        pair = _extract_pair(crop[:, lx], settings, "top", "bottom")
        if pair is None:
            continue
        top_y, bottom_y = float(y1 + pair[0]), float(y1 + pair[1])
        values.append(bottom_y - top_y)
        pairs.append((float(x1 + lx), top_y, bottom_y))
    return _finish(result, values, pairs, settings)
