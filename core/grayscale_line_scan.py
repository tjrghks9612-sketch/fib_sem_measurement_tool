from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from fib_sem_measurement_tool.models.result import EdgeScanResult, PairCandidate, RawEdgeCandidate
from fib_sem_measurement_tool.models.settings import MeasurementSettings


SOBEL_KERNEL_SIZE = 3
SOBEL_DELTA_SCALE = 4.0
DEFAULT_MAX_JUMP_PX = 28.0
DEFAULT_MAX_PROFILE_CANDIDATES_PER_SCANLINE = 16
PROJECTION_PERCENTILE = 75.0


def _robust_normalize_signal(data: np.ndarray, settings: MeasurementSettings) -> np.ndarray:
    if not bool(getattr(settings, "normalize_grayscale_profiles", False)):
        return data.astype(np.float32, copy=True)

    low_pct = float(getattr(settings, "normalize_low_percentile", 2.0))
    high_pct = float(getattr(settings, "normalize_high_percentile", 98.0))
    if high_pct <= low_pct:
        return data.astype(np.float32, copy=True)

    low = float(np.percentile(data, low_pct))
    high = float(np.percentile(data, high_pct))
    span = high - low
    min_span = max(1.0, float(getattr(settings, "normalize_min_span", 12.0)))
    if span < min_span:
        return data.astype(np.float32, copy=True)
    return np.clip((data.astype(np.float32, copy=False) - low) * (255.0 / span), 0.0, 255.0).astype(np.float32)


def _median_filter_along_scan_axis(data: np.ndarray, orientation: str, window: int) -> np.ndarray:
    win = max(1, int(window))
    if win <= 1:
        return data.astype(np.float32, copy=True)
    if win % 2 == 0:
        win += 1

    axis = 1 if orientation == "horizontal" else 0
    if data.shape[axis] < win:
        return data.astype(np.float32, copy=True)

    pad = win // 2
    pad_width = ((0, 0), (pad, pad)) if axis == 1 else ((pad, pad), (0, 0))
    padded = np.pad(data, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=win, axis=axis)
    return np.median(windows, axis=-1).astype(np.float32)


def _edge_preserving_smooth_along_scan_axis(
    data: np.ndarray,
    orientation: str,
    window: int,
    range_sigma: float,
) -> np.ndarray:
    win = max(1, int(window))
    if win <= 1:
        return data.astype(np.float32, copy=True)
    if win % 2 == 0:
        win += 1

    axis = 1 if orientation == "horizontal" else 0
    if data.shape[axis] < 3:
        return data.astype(np.float32, copy=True)

    source = data.astype(np.float32, copy=False)
    radius = min(win // 2, max(1, data.shape[axis] - 1))
    sigma_space = max(1.0, float(radius) / 2.0)
    sigma_range = max(1.0, float(range_sigma))
    weighted_sum = source.copy()
    weight_sum = np.ones_like(source, dtype=np.float32)

    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        shifted = np.roll(source, shift=offset, axis=axis)
        if axis == 1:
            if offset < 0:
                shifted[:, offset:] = source[:, offset:]
            else:
                shifted[:, :offset] = source[:, :offset]
        else:
            if offset < 0:
                shifted[offset:, :] = source[offset:, :]
            else:
                shifted[:offset, :] = source[:offset, :]

        spatial_weight = np.exp(-float(offset * offset) / (2.0 * sigma_space * sigma_space))
        range_weight = np.exp(-((shifted - source) ** 2) / (2.0 * sigma_range * sigma_range))
        weight = (spatial_weight * range_weight).astype(np.float32)
        weighted_sum += shifted * weight
        weight_sum += weight

    return (weighted_sum / np.maximum(weight_sum, 1e-6)).astype(np.float32)


def _prepare_detection_crop(crop: np.ndarray, orientation: str, settings: MeasurementSettings) -> np.ndarray:
    signal = _robust_normalize_signal(crop, settings)
    if bool(getattr(settings, "denoise_grayscale_profiles", False)):
        signal = _edge_preserving_smooth_along_scan_axis(
            signal,
            orientation,
            int(getattr(settings, "profile_denoise_window", 3)),
            float(getattr(settings, "profile_denoise_range_sigma", 28.0)),
        )
    return signal


def prepare_display_profile_signal(
    profile: Sequence[float],
    orientation: str,
    settings: MeasurementSettings,
) -> np.ndarray:
    data = np.asarray(profile, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    if orientation == "horizontal":
        crop = data.reshape(1, -1)
    elif orientation == "vertical":
        crop = data.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported profile orientation: {orientation}")
    return _prepare_detection_crop(crop, orientation, settings).reshape(-1).astype(np.float32)


def prepare_display_profile_from_roi(
    gray: np.ndarray,
    roi: Sequence[int],
    orientation: str,
    scan_index: int,
    settings: MeasurementSettings,
) -> tuple[np.ndarray, int, int]:
    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = np.asarray(gray[y1 : y2 + 1, x1 : x2 + 1], dtype=np.float32)
    if crop.size == 0:
        return np.asarray([], dtype=np.float32), 0, int(scan_index)
    detection_crop = _prepare_detection_crop(crop, orientation, settings)
    if orientation == "horizontal":
        local_y = max(0, min(detection_crop.shape[0] - 1, int(scan_index) - y1))
        return detection_crop[local_y, :].reshape(-1).astype(np.float32), x1, y1 + local_y
    if orientation == "vertical":
        local_x = max(0, min(detection_crop.shape[1] - 1, int(scan_index) - x1))
        return detection_crop[:, local_x].reshape(-1).astype(np.float32), y1, x1 + local_x
    raise ValueError(f"Unsupported profile orientation: {orientation}")


def _minimum_delta(settings: MeasurementSettings) -> float:
    return max(0.0, float(getattr(settings, "minimum_grayscale_delta", 30.0)))


def _max_profile_candidates(settings: MeasurementSettings) -> int:
    return max(
        0,
        int(
            getattr(
                settings,
                "max_profile_candidates_per_scanline",
                DEFAULT_MAX_PROFILE_CANDIDATES_PER_SCANLINE,
            )
        ),
    )


def _candidate_image_position(
    scan_axis: str,
    scan_index: int,
    position: float,
    roi_origin: tuple[int, int],
) -> tuple[float, float]:
    x0, y0 = roi_origin
    if scan_axis == "horizontal":
        return float(x0 + position), float(scan_index)
    if scan_axis == "vertical":
        return float(scan_index), float(y0 + position)
    raise ValueError(f"Unsupported scan axis: {scan_axis}")


def _sobel_gradient_profile(data: np.ndarray, scan_axis: str) -> np.ndarray:
    if scan_axis == "horizontal":
        gradient = cv2.Sobel(data.reshape(1, -1), cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        return gradient.reshape(-1) / SOBEL_DELTA_SCALE
    if scan_axis == "vertical":
        gradient = cv2.Sobel(data.reshape(-1, 1), cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
        return gradient.reshape(-1) / SOBEL_DELTA_SCALE
    raise ValueError(f"Unsupported scan axis: {scan_axis}")


def _sobel_gradient_image(crop: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        return cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE) / SOBEL_DELTA_SCALE
    if orientation == "vertical":
        return cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE) / SOBEL_DELTA_SCALE
    raise ValueError(f"Unsupported scan orientation: {orientation}")


def _gradient_projection(gradient: np.ndarray, orientation: str) -> np.ndarray:
    strength = np.abs(np.asarray(gradient, dtype=np.float32))
    if strength.size == 0:
        return np.asarray([], dtype=np.float64)
    if orientation == "horizontal":
        return np.percentile(strength, PROJECTION_PERCENTILE, axis=0).astype(np.float64)
    if orientation == "vertical":
        return np.percentile(strength, PROJECTION_PERCENTILE, axis=1).astype(np.float64)
    raise ValueError(f"Unsupported scan orientation: {orientation}")


def _gradient_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _local_peak_runs(gradient: np.ndarray, min_delta: float) -> List[tuple[int, int]]:
    strength = np.abs(gradient)
    if strength.size == 0:
        return []

    peaks: List[tuple[int, int]] = []
    eps = 1e-6
    idx = 0
    while idx < strength.size:
        value = float(strength[idx])
        sign = _gradient_sign(float(gradient[idx]))
        if not np.isfinite(value) or value < min_delta or value <= 0.0 or sign == 0:
            idx += 1
            continue

        start = idx
        end = idx
        while (
            end + 1 < strength.size
            and _gradient_sign(float(gradient[end + 1])) == sign
            and abs(float(strength[end + 1]) - value) <= eps
        ):
            end += 1

        left = (
            float(strength[start - 1])
            if start > 0 and _gradient_sign(float(gradient[start - 1])) == sign
            else -np.inf
        )
        right = (
            float(strength[end + 1])
            if end + 1 < strength.size and _gradient_sign(float(gradient[end + 1])) == sign
            else -np.inf
        )
        if value > left + eps and value > right + eps:
            peaks.append((start, end))
        idx = end + 1
    return peaks


def _peak_position(start: int, end: int, signed_delta: float, data: Optional[np.ndarray] = None) -> float:
    if data is not None and data.size >= 2:
        raw_diff = data[1:] - data[:-1]
        sign = _gradient_sign(signed_delta)
        first = max(0, start - 1)
        last = min(raw_diff.size - 1, end)
        transition_indices = [
            index
            for index in range(first, last + 1)
            if _gradient_sign(float(raw_diff[index])) == sign
        ]
        if transition_indices:
            best = max(transition_indices, key=lambda index: abs(float(raw_diff[index])))
            return float(best) + 0.5

    center = (float(start) + float(end)) / 2.0
    if start != end:
        return center
    if signed_delta > 0.0:
        return center + 0.5
    if signed_delta < 0.0:
        return center - 0.5
    return center


def _profile_transition_values(data: np.ndarray, position: float) -> tuple[float, float]:
    before_idx = int(np.floor(position))
    after_idx = before_idx + 1
    before_idx = max(0, min(before_idx, data.size - 1))
    after_idx = max(0, min(after_idx, data.size - 1))
    return float(data[before_idx]), float(data[after_idx])


def detect_profile_candidates(
    profile: Sequence[float],
    settings: MeasurementSettings,
    scan_axis: str,
    scan_index: int,
    local_scan_index: int,
    roi_origin: tuple[int, int],
    gradient_profile: Optional[Sequence[float]] = None,
    detection_profile: Optional[Sequence[float]] = None,
) -> List[RawEdgeCandidate]:
    data = np.asarray(profile, dtype=np.float32).reshape(-1)
    if data.size < 2:
        return []
    detection_data = (
        np.asarray(detection_profile, dtype=np.float32).reshape(-1)
        if detection_profile is not None
        else data
    )
    if detection_data.size != data.size:
        raise ValueError("detection_profile must have the same length as profile")

    if gradient_profile is None:
        gradient = _sobel_gradient_profile(detection_data, scan_axis)
    else:
        gradient = np.asarray(gradient_profile, dtype=np.float32).reshape(-1)
        if gradient.size != data.size:
            raise ValueError("gradient_profile must have the same length as profile")
    min_delta = _minimum_delta(settings)

    candidates: List[RawEdgeCandidate] = []
    for start, end in _local_peak_runs(gradient, min_delta):
        signed_values = gradient[start : end + 1]
        delta = float(np.mean(signed_values, dtype=np.float64))
        if delta == 0.0:
            continue
        position = _peak_position(start, end, delta, detection_data)
        position = max(0.0, min(float(data.size - 1), position))
        grayscale_before, grayscale_after = _profile_transition_values(data, position)
        image_x, image_y = _candidate_image_position(scan_axis, scan_index, position, roi_origin)
        candidates.append(
            RawEdgeCandidate(
                scan_axis=scan_axis,
                scan_index=int(scan_index),
                local_scan_index=int(local_scan_index),
                position=position,
                image_x=image_x,
                image_y=image_y,
                strength=float(abs(delta)),
                signed_delta=delta,
                sign=1 if delta > 0.0 else -1,
                grayscale_before=grayscale_before,
                grayscale_after=grayscale_after,
            )
        )
    limit = _max_profile_candidates(settings)
    if limit > 0 and len(candidates) > limit:
        candidates = sorted(candidates, key=lambda candidate: candidate.strength, reverse=True)[:limit]
        candidates.sort(key=lambda candidate: candidate.position)
    return candidates


def scan_raw_edge_candidates(
    gray: np.ndarray,
    roi: Sequence[int],
    orientation: str,
    settings: MeasurementSettings,
) -> EdgeScanResult:
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(f"Unsupported scan orientation: {orientation}")

    x1, y1, x2, y2 = [int(v) for v in roi]
    crop = np.asarray(gray[y1 : y2 + 1, x1 : x2 + 1], dtype=np.float32)
    detection_crop = _prepare_detection_crop(crop, orientation, settings)
    gradient = _sobel_gradient_image(detection_crop, orientation)
    gradient_projection = _gradient_projection(gradient, orientation)
    scanned_line_count = int(crop.shape[0] if orientation == "horizontal" else crop.shape[1])
    per_scanline_candidate_count: Dict[int, int] = {}
    profiles_by_scanline: Dict[int, List[float]] = {}
    raw_candidates: List[RawEdgeCandidate] = []

    if orientation == "horizontal":
        for local_y in range(crop.shape[0]):
            scan_index = y1 + local_y
            profile = crop[local_y, :]
            profiles_by_scanline[int(scan_index)] = [float(value) for value in profile]
            candidates = detect_profile_candidates(
                profile,
                settings,
                scan_axis="horizontal",
                scan_index=scan_index,
                local_scan_index=local_y,
                roi_origin=(x1, y1),
                gradient_profile=gradient[local_y, :],
                detection_profile=detection_crop[local_y, :],
            )
            per_scanline_candidate_count[int(scan_index)] = len(candidates)
            raw_candidates.extend(candidates)
    else:
        for local_x in range(crop.shape[1]):
            scan_index = x1 + local_x
            profile = crop[:, local_x]
            profiles_by_scanline[int(scan_index)] = [float(value) for value in profile]
            candidates = detect_profile_candidates(
                profile,
                settings,
                scan_axis="vertical",
                scan_index=scan_index,
                local_scan_index=local_x,
                roi_origin=(x1, y1),
                gradient_profile=gradient[:, local_x],
                detection_profile=detection_crop[:, local_x],
            )
            per_scanline_candidate_count[int(scan_index)] = len(candidates)
            raw_candidates.extend(candidates)

    return EdgeScanResult(
        scan_axis=orientation,
        roi=(x1, y1, x2, y2),
        scanned_line_count=scanned_line_count,
        raw_edge_candidates=raw_candidates,
        per_scanline_candidate_count=per_scanline_candidate_count,
        profiles_by_scanline=profiles_by_scanline,
        gradient_projection=[float(value) for value in gradient_projection],
        minimum_grayscale_delta=_minimum_delta(settings),
    )


def _group_by_scanline(candidates: Iterable[RawEdgeCandidate]) -> Dict[int, List[RawEdgeCandidate]]:
    grouped: Dict[int, List[RawEdgeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate.scan_index)].append(candidate)
    for items in grouped.values():
        items.sort(key=lambda candidate: candidate.position)
    return dict(grouped)


def scan_pair_candidates(scan_result: EdgeScanResult) -> List[PairCandidate]:
    pairs: List[PairCandidate] = []
    for scan_index, candidates in _group_by_scanline(scan_result.raw_edge_candidates).items():
        for first, second in combinations(candidates, 2):
            if second.position <= first.position:
                continue
            pairs.append(
                PairCandidate(
                    scan_axis=scan_result.scan_axis,
                    scan_index=int(scan_index),
                    local_scan_index=int(first.local_scan_index),
                    first=first,
                    second=second,
                )
            )
    return pairs


def _scan_profile_length(scan_result: EdgeScanResult) -> int:
    x1, y1, x2, y2 = scan_result.roi
    if scan_result.scan_axis == "horizontal":
        return int(x2 - x1 + 1)
    return int(y2 - y1 + 1)


def _side_pool(candidates: Sequence[RawEdgeCandidate], midpoint: float, side: str) -> List[RawEdgeCandidate]:
    if side in {"left", "top", "first"}:
        return [candidate for candidate in candidates if candidate.position <= midpoint]
    if side in {"right", "bottom", "second"}:
        return [candidate for candidate in candidates if candidate.position > midpoint]
    raise ValueError(f"Unsupported boundary side: {side}")


def _default_direction(scan_axis: str, side: str) -> str:
    if scan_axis == "horizontal":
        return "left_to_center" if side in {"left", "first"} else "right_to_center"
    if scan_axis == "vertical":
        return "top_to_bottom" if side in {"top", "first"} else "bottom_to_top"
    raise ValueError(f"Unsupported scan axis: {scan_axis}")


def scan_edge_in_direction(
    candidates: Sequence[RawEdgeCandidate],
    direction: str,
    from_left: bool = True,
    prefer_strongest: bool = False,
) -> Optional[RawEdgeCandidate]:
    if not candidates:
        return None

    if direction == "outside_to_center":
        ordered = sorted(candidates, key=lambda candidate: candidate.position, reverse=not from_left)
    elif direction == "center_to_outside":
        ordered = sorted(candidates, key=lambda candidate: candidate.position, reverse=from_left)
    elif direction in {"left_to_center", "center_to_right", "top_to_bottom"}:
        ordered = sorted(candidates, key=lambda candidate: candidate.position)
    elif direction in {"center_to_left", "right_to_center", "bottom_to_top"}:
        ordered = sorted(candidates, key=lambda candidate: candidate.position, reverse=True)
    else:
        raise ValueError(f"Unsupported edge scan direction: {direction}")

    if not prefer_strongest:
        return ordered[0]

    rank = {id(candidate): index for index, candidate in enumerate(ordered)}
    return max(
        ordered,
        key=lambda candidate: (
            float(candidate.strength),
            abs(float(candidate.signed_delta)),
            -rank[id(candidate)],
        ),
    )


def _preferred_sign(side: str) -> int:
    return -1 if side in {"left", "top", "first"} else 1


def _strongest_candidate(candidates: Sequence[RawEdgeCandidate], side: str) -> Optional[RawEdgeCandidate]:
    if not candidates:
        return None
    preferred = [candidate for candidate in candidates if candidate.sign == _preferred_sign(side)]
    pool = preferred or list(candidates)
    return max(pool, key=lambda candidate: (candidate.strength, -abs(candidate.signed_delta), candidate.position))


def select_strongest_boundary_candidates(scan_result: EdgeScanResult, side: str) -> List[RawEdgeCandidate]:
    midpoint = float(_scan_profile_length(scan_result) - 1) / 2.0
    selected: List[RawEdgeCandidate] = []
    for scan_index, candidates in _group_by_scanline(scan_result.raw_edge_candidates).items():
        side_candidates = _side_pool(candidates, midpoint, side)
        candidate = _strongest_candidate(side_candidates, side)
        if candidate is not None:
            selected.append(candidate)
    return selected


def select_first_valid_boundary_candidates(
    scan_result: EdgeScanResult,
    side: str,
    direction: Optional[str] = None,
    prefer_sign: bool = False,
) -> List[RawEdgeCandidate]:
    midpoint = float(_scan_profile_length(scan_result) - 1) / 2.0
    scan_direction = direction or _default_direction(scan_result.scan_axis, side)
    selected: List[RawEdgeCandidate] = []
    for _scan_index, candidates in _group_by_scanline(scan_result.raw_edge_candidates).items():
        side_candidates = _side_pool(candidates, midpoint, side)
        if prefer_sign:
            preferred = [candidate for candidate in side_candidates if candidate.sign == _preferred_sign(side)]
            side_candidates = preferred or side_candidates
        candidate = scan_edge_in_direction(
            side_candidates,
            scan_direction,
            from_left=side in {"left", "top", "first"},
        )
        if candidate is not None:
            selected.append(candidate)
    return selected


def _profile_coordinate(candidate: RawEdgeCandidate) -> float:
    if candidate.scan_axis == "horizontal":
        return float(candidate.image_x)
    return float(candidate.image_y)


def refine_boundary_candidates_by_line(
    candidates: Sequence[RawEdgeCandidate],
    residual_limit_px: float = 6.0,
    mad_multiplier: float = 2.5,
    iterations: int = 3,
) -> List[RawEdgeCandidate]:
    selected = list(candidates)
    if len(selected) < 4:
        return selected

    for _ in range(max(1, int(iterations))):
        if len(selected) < 4:
            break
        scan_indices = np.asarray([candidate.scan_index for candidate in selected], dtype=np.float64)
        coordinates = np.asarray([_profile_coordinate(candidate) for candidate in selected], dtype=np.float64)
        slope, intercept = np.polyfit(scan_indices, coordinates, 1)
        residuals = coordinates - (slope * scan_indices + intercept)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        limit = max(float(residual_limit_px), float(mad_multiplier) * 1.4826 * (mad + 1e-6))
        keep = np.abs(residuals - median) <= limit
        if bool(np.all(keep)):
            break
        selected = [candidate for candidate, is_kept in zip(selected, keep) if bool(is_kept)]
    return sorted(selected, key=lambda candidate: candidate.scan_index)


def filter_boundary_candidates_by_continuity(
    candidates: Sequence[RawEdgeCandidate],
    window_size: int = 5,
    max_jump_px: float = 6.0,
    mad_multiplier: float = 3.0,
) -> List[RawEdgeCandidate]:
    selected = sorted(candidates, key=lambda candidate: candidate.scan_index)
    if len(selected) < 4:
        return selected

    coordinates = np.asarray([_profile_coordinate(candidate) for candidate in selected], dtype=np.float64)
    half_window = max(1, int(window_size) // 2)
    keep: List[bool] = []
    for idx, coordinate in enumerate(coordinates):
        start = max(0, idx - half_window)
        end = min(len(coordinates), idx + half_window + 1)
        local = coordinates[start:end]
        local_median = float(np.median(local))
        local_mad = float(np.median(np.abs(local - local_median)))
        limit = max(float(max_jump_px), float(mad_multiplier) * 1.4826 * (local_mad + 1e-6))
        keep.append(abs(float(coordinate) - local_median) <= limit)
    return [candidate for candidate, is_kept in zip(selected, keep) if is_kept]


def fill_small_gaps_nan(values: Sequence[float], max_gap: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).copy()
    if data.size == 0:
        return data

    limit = max(0, int(max_gap))
    isnan = np.isnan(data)
    idx = 0
    while idx < data.size:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < data.size and isnan[idx]:
            idx += 1
        end = idx
        gap = end - start
        left = start - 1
        right = end
        if gap <= limit and left >= 0 and right < data.size and not np.isnan(data[left]) and not np.isnan(data[right]):
            data[start:end] = np.linspace(data[left], data[right], gap + 2, dtype=np.float64)[1:-1]
    return data


def moving_average_nan(values: Sequence[float], win: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return data.copy()

    window = max(1, int(win))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = np.full(data.shape, np.nan, dtype=np.float64)
    for idx, value in enumerate(data):
        if np.isnan(value):
            continue
        start = max(0, idx - half)
        end = min(data.size, idx + half + 1)
        local = data[start:end]
        finite = local[np.isfinite(local)]
        if finite.size:
            smoothed[idx] = float(np.mean(finite))
    return smoothed


def _scan_indices(scan_result: EdgeScanResult) -> List[int]:
    x1, y1, x2, y2 = scan_result.roi
    if scan_result.scan_axis == "horizontal":
        return list(range(int(y1), int(y2) + 1))
    return list(range(int(x1), int(x2) + 1))


def _local_scan_index(scan_result: EdgeScanResult, scan_index: int) -> int:
    x1, y1, _x2, _y2 = scan_result.roi
    if scan_result.scan_axis == "horizontal":
        return int(scan_index) - int(y1)
    return int(scan_index) - int(x1)


def _candidate_from_position(
    scan_result: EdgeScanResult,
    scan_index: int,
    position: float,
    template: Optional[RawEdgeCandidate] = None,
) -> RawEdgeCandidate:
    x1, y1, _x2, _y2 = scan_result.roi
    profile = _scanline_profile(scan_result, scan_index)
    if profile is not None:
        grayscale_before, grayscale_after = _profile_transition_values(profile, position)
    elif template is not None:
        grayscale_before = float(template.grayscale_before)
        grayscale_after = float(template.grayscale_after)
    else:
        grayscale_before = 0.0
        grayscale_after = 0.0

    if template is not None:
        signed_delta = float(template.signed_delta)
        strength = float(template.strength)
        sign = int(template.sign)
    else:
        signed_delta = float(grayscale_after - grayscale_before)
        strength = abs(signed_delta)
        sign = 1 if signed_delta >= 0.0 else -1

    image_x, image_y = _candidate_image_position(scan_result.scan_axis, int(scan_index), float(position), (x1, y1))
    return RawEdgeCandidate(
        scan_axis=scan_result.scan_axis,
        scan_index=int(scan_index),
        local_scan_index=_local_scan_index(scan_result, int(scan_index)),
        position=float(position),
        image_x=image_x,
        image_y=image_y,
        strength=strength,
        signed_delta=signed_delta,
        sign=sign,
        grayscale_before=grayscale_before,
        grayscale_after=grayscale_after,
    )


def postprocess_boundary_candidates(
    scan_result: EdgeScanResult,
    candidates: Sequence[RawEdgeCandidate],
    max_gap: int,
    smooth_window: int,
) -> List[RawEdgeCandidate]:
    scan_indices = _scan_indices(scan_result)
    if not scan_indices:
        return []

    by_scan: Dict[int, RawEdgeCandidate] = {}
    for candidate in candidates:
        existing = by_scan.get(int(candidate.scan_index))
        if existing is None or candidate.strength > existing.strength:
            by_scan[int(candidate.scan_index)] = candidate

    values = np.full(len(scan_indices), np.nan, dtype=np.float64)
    for idx, scan_index in enumerate(scan_indices):
        candidate = by_scan.get(int(scan_index))
        if candidate is not None:
            values[idx] = float(candidate.position)

    filled = fill_small_gaps_nan(values, max_gap=max_gap)
    smoothed = moving_average_nan(filled, win=smooth_window)
    profile_length = _scan_profile_length(scan_result)
    result: List[RawEdgeCandidate] = []
    for idx, position in enumerate(smoothed):
        if not np.isfinite(position):
            continue
        scan_index = scan_indices[idx]
        template = by_scan.get(int(scan_index))
        if template is None:
            template = min(
                by_scan.values(),
                key=lambda candidate: abs(int(candidate.scan_index) - int(scan_index)),
                default=None,
            )
        clean_position = max(0.0, min(float(profile_length - 1), float(position)))
        result.append(_candidate_from_position(scan_result, scan_index, clean_position, template))
    return result


def _repair_boundary_jumps(
    scan_result: EdgeScanResult,
    candidates: Sequence[RawEdgeCandidate],
    side: str,
    max_jump_px: float,
) -> List[RawEdgeCandidate]:
    selected = sorted(candidates, key=lambda candidate: candidate.scan_index)
    if not selected:
        return []

    grouped = _group_by_scanline(scan_result.raw_edge_candidates)
    midpoint = float(_scan_profile_length(scan_result) - 1) / 2.0
    repaired: List[RawEdgeCandidate] = []
    previous_position: Optional[float] = None
    jump_limit = max(0.0, float(max_jump_px))
    for candidate in selected:
        chosen = candidate
        if previous_position is not None and abs(float(candidate.position) - previous_position) > jump_limit:
            same_line = _side_pool(grouped.get(int(candidate.scan_index), []), midpoint, side)
            nearby = [
                item
                for item in same_line
                if abs(float(item.position) - previous_position) <= jump_limit
            ]
            if nearby:
                chosen = min(
                    nearby,
                    key=lambda item: (
                        abs(float(item.position) - previous_position),
                        -float(item.strength),
                    ),
                )
        repaired.append(chosen)
        previous_position = float(chosen.position)
    return repaired


def _postprocess_defaults(scan_axis: str) -> tuple[int, int]:
    if scan_axis == "horizontal":
        return 10, 7
    if scan_axis == "vertical":
        return 6, 3
    raise ValueError(f"Unsupported scan axis: {scan_axis}")


def _pairs_from_boundary_candidates(
    scan_result: EdgeScanResult,
    first_candidates: Sequence[RawEdgeCandidate],
    second_candidates: Sequence[RawEdgeCandidate],
) -> List[PairCandidate]:
    first_by_scan = {int(candidate.scan_index): candidate for candidate in first_candidates}
    second_by_scan = {int(candidate.scan_index): candidate for candidate in second_candidates}
    selected: List[PairCandidate] = []
    for scan_index in sorted(set(first_by_scan) & set(second_by_scan)):
        first = first_by_scan[scan_index]
        second = second_by_scan[scan_index]
        if second.position <= first.position:
            continue
        selected.append(
            PairCandidate(
                scan_axis=scan_result.scan_axis,
                scan_index=int(scan_index),
                local_scan_index=int(first.local_scan_index),
                first=first,
                second=second,
            )
        )
    return selected


def _candidate_density_clusters(
    scan_result: EdgeScanResult,
    min_coverage: float = 0.12,
    smooth_window: int = 5,
) -> List[dict[str, float]]:
    profile_length = _scan_profile_length(scan_result)
    if profile_length <= 0 or scan_result.scanned_line_count <= 0:
        return []

    counts = np.zeros(profile_length, dtype=np.float64)
    for candidate in scan_result.raw_edge_candidates:
        position = int(round(float(candidate.position)))
        if 0 <= position < profile_length:
            counts[position] += 1.0

    window = max(1, int(smooth_window))
    if window % 2 == 0:
        window += 1
    smoothed = np.convolve(counts, np.ones(window, dtype=np.float64) / float(window), mode="same")
    threshold = max(8.0, float(scan_result.scanned_line_count) * float(min_coverage))
    active = smoothed >= threshold

    clusters: List[dict[str, float]] = []
    idx = 0
    while idx < profile_length:
        if not bool(active[idx]):
            idx += 1
            continue
        start = idx
        while idx < profile_length and bool(active[idx]):
            idx += 1
        end = idx - 1
        margin = max(2, window // 2)
        low = max(0.0, float(start - margin))
        high = min(float(profile_length - 1), float(end + margin))
        members = [
            candidate
            for candidate in scan_result.raw_edge_candidates
            if low <= float(candidate.position) <= high
        ]
        if not members:
            continue
        unique_scanlines = {int(candidate.scan_index) for candidate in members}
        coverage = float(len(unique_scanlines) / scan_result.scanned_line_count)
        if coverage < float(min_coverage):
            continue
        strengths = np.asarray([candidate.strength for candidate in members], dtype=np.float64)
        positions = np.asarray([candidate.position for candidate in members], dtype=np.float64)
        weight_sum = float(np.sum(strengths))
        center = float(np.average(positions, weights=strengths)) if weight_sum > 0.0 else float(np.median(positions))
        clusters.append(
            {
                "start": low,
                "end": high,
                "center": center,
                "coverage": coverage,
                "median_strength": float(np.median(strengths)),
                "score": float(coverage * max(1.0, float(np.median(strengths)))),
            }
        )
    return sorted(clusters, key=lambda cluster: cluster["center"])


def _candidate_near_cluster(
    candidates: Sequence[RawEdgeCandidate],
    cluster: dict[str, float],
    margin_px: float,
) -> Optional[RawEdgeCandidate]:
    low = float(cluster["start"]) - float(margin_px)
    high = float(cluster["end"]) + float(margin_px)
    center = float(cluster["center"])
    pool = [candidate for candidate in candidates if low <= float(candidate.position) <= high]
    if not pool:
        return None
    return max(
        pool,
        key=lambda candidate: (
            float(candidate.strength),
            -abs(float(candidate.position) - center),
        ),
    )


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return values.astype(np.float64, copy=True)
    return np.convolve(values.astype(np.float64, copy=False), np.ones(win, dtype=np.float64) / float(win), mode="same")


def _projection_peak_clusters(
    scan_result: EdgeScanResult,
    max_peaks: int = 12,
) -> List[dict[str, float]]:
    projection = np.asarray(scan_result.gradient_projection, dtype=np.float64).reshape(-1)
    profile_length = _scan_profile_length(scan_result)
    if projection.size != profile_length or profile_length < 3:
        return []

    smooth_window = max(5, min(15, int(round(float(profile_length) * 0.03))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = _smooth_1d(projection, smooth_window)
    max_value = float(np.max(smoothed)) if smoothed.size else 0.0
    if max_value <= 0.0:
        return []

    baseline = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - baseline)))
    threshold = max(baseline + 2.5 * 1.4826 * (mad + 1e-6), max_value * 0.25)
    candidates: List[tuple[float, int]] = []
    for idx in range(1, smoothed.size - 1):
        value = float(smoothed[idx])
        if value < threshold:
            continue
        if value > float(smoothed[idx - 1]) and value >= float(smoothed[idx + 1]):
            candidates.append((value, idx))
    if not candidates:
        return []

    min_peak_distance = max(6, int(round(float(profile_length) * 0.06)))
    selected: List[tuple[float, int]] = []
    for value, idx in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(abs(idx - other_idx) < min_peak_distance for _other_value, other_idx in selected):
            continue
        selected.append((value, idx))
        if len(selected) >= max_peaks:
            break

    half_width = max(2.0, float(min_peak_distance) / 2.0)
    clusters = [
        {
            "start": max(0.0, float(idx) - half_width),
            "end": min(float(profile_length - 1), float(idx) + half_width),
            "center": float(idx),
            "coverage": 1.0,
            "median_strength": float(value),
            "score": float(value),
        }
        for value, idx in selected
    ]
    return sorted(clusters, key=lambda cluster: cluster["center"])


def _select_projection_layer_clusters(scan_result: EdgeScanResult) -> Optional[tuple[dict[str, float], dict[str, float]]]:
    clusters = _projection_peak_clusters(scan_result)
    if len(clusters) < 2:
        return None

    profile_length = _scan_profile_length(scan_result)
    min_separation = max(12.0, float(profile_length) * 0.08)
    best_pair: Optional[tuple[dict[str, float], dict[str, float]]] = None
    best_score: Optional[tuple[float, float, float]] = None
    for first, second in combinations(clusters, 2):
        separation = float(second["center"]) - float(first["center"])
        if separation < min_separation:
            continue
        first_score = float(first["score"])
        second_score = float(second["score"])
        score = (
            min(first_score, second_score) * 3.0
            + (first_score + second_score) * 0.25
            + separation * 0.02
        )
        key = (score, min(first_score, second_score), -separation)
        if best_score is None or key > best_score:
            best_score = key
            best_pair = (first, second)
    return best_pair


def select_projection_layer_pairs_per_scanline(
    scan_result: EdgeScanResult,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    min_coverage: float = 0.45,
) -> List[PairCandidate]:
    if scan_result.scan_axis != "vertical" or scan_result.scanned_line_count < 128:
        return []

    cluster_pair = _select_projection_layer_clusters(scan_result)
    if cluster_pair is None:
        return []
    first_cluster, second_cluster = cluster_pair

    grouped = _group_by_scanline(scan_result.raw_edge_candidates)
    profile_length = _scan_profile_length(scan_result)
    cluster_margin = max(5.0, float(profile_length) * 0.03)
    first_candidates: List[RawEdgeCandidate] = []
    second_candidates: List[RawEdgeCandidate] = []
    for scan_index in _scan_indices(scan_result):
        candidates = grouped.get(int(scan_index), [])
        first = _candidate_near_cluster(candidates, first_cluster, cluster_margin)
        second = _candidate_near_cluster(candidates, second_cluster, cluster_margin)
        if first is None or second is None or second.position <= first.position:
            continue
        first_candidates.append(first)
        second_candidates.append(second)

    selected_coverage = float(len(first_candidates) / scan_result.scanned_line_count) if scan_result.scanned_line_count else 0.0
    if selected_coverage < float(min_coverage):
        return []

    max_gap, smooth_window = _postprocess_defaults(scan_result.scan_axis)
    first = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, first_candidates, "first", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    second = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, second_candidates, "second", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    return _pairs_from_boundary_candidates(scan_result, first, second)


def select_density_layer_pairs_per_scanline(
    scan_result: EdgeScanResult,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    min_coverage: float = 0.12,
) -> List[PairCandidate]:
    if scan_result.scan_axis != "vertical" or scan_result.scanned_line_count < 128:
        return []

    clusters = _candidate_density_clusters(scan_result, min_coverage=min_coverage)
    if len(clusters) < 2:
        return []

    profile_length = _scan_profile_length(scan_result)
    min_separation = max(12.0, float(profile_length) * 0.08)
    first_cluster = clusters[0]
    second_cluster: Optional[dict[str, float]] = None
    for cluster in clusters[1:]:
        if float(cluster["center"]) - float(first_cluster["center"]) < min_separation:
            continue
        second_cluster = cluster
        break
    if second_cluster is None:
        return []

    grouped = _group_by_scanline(scan_result.raw_edge_candidates)
    first_candidates: List[RawEdgeCandidate] = []
    second_candidates: List[RawEdgeCandidate] = []
    cluster_margin = max(4.0, float(profile_length) * 0.025)
    for scan_index in _scan_indices(scan_result):
        candidates = grouped.get(int(scan_index), [])
        first = _candidate_near_cluster(candidates, first_cluster, cluster_margin)
        second = _candidate_near_cluster(candidates, second_cluster, cluster_margin)
        if first is None or second is None or second.position <= first.position:
            continue
        first_candidates.append(first)
        second_candidates.append(second)

    if not first_candidates or not second_candidates:
        return []

    max_gap, smooth_window = _postprocess_defaults(scan_result.scan_axis)
    first = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, first_candidates, "first", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    second = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, second_candidates, "second", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    return _pairs_from_boundary_candidates(scan_result, first, second)


def _scanline_profile(scan_result: EdgeScanResult, scan_index: int) -> Optional[np.ndarray]:
    profile = scan_result.profiles_by_scanline.get(int(scan_index))
    if profile is None:
        return None
    data = np.asarray(profile, dtype=np.float32).reshape(-1)
    if data.size < 2:
        return None
    return data


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values.astype(np.float64, copy=False)))


def _edge_window_size(profile_length: int) -> int:
    return max(3, min(12, int(round(float(profile_length) * 0.04))))


def _stable_region_score(
    profile: np.ndarray,
    first: RawEdgeCandidate,
    second: RawEdgeCandidate,
    minimum_delta: float,
) -> Optional[tuple[float, float, float, float]]:
    first_index = int(np.floor(first.position))
    second_index = int(np.floor(second.position))
    if first_index < 0 or second_index + 1 >= profile.size or second_index <= first_index:
        return None

    inside_start = first_index + 1
    inside_end = second_index + 1
    if inside_end <= inside_start:
        return None

    window = _edge_window_size(int(profile.size))
    left_outer = profile[max(0, first_index - window + 1) : first_index + 1]
    right_outer = profile[second_index + 1 : min(profile.size, second_index + 1 + window)]
    inside = profile[inside_start:inside_end]
    left_inner = profile[inside_start : min(inside_end, inside_start + window)]
    right_inner = profile[max(inside_start, inside_end - window) : inside_end]
    if left_outer.size == 0 or right_outer.size == 0 or left_inner.size == 0 or right_inner.size == 0:
        return None

    inside_level = _median(inside)
    left_outer_level = _median(left_outer)
    right_outer_level = _median(right_outer)
    left_inner_level = _median(left_inner)
    right_inner_level = _median(right_inner)

    left_region_contrast = abs(inside_level - left_outer_level)
    right_region_contrast = abs(right_outer_level - inside_level)
    left_edge_contrast = abs(left_inner_level - left_outer_level)
    right_edge_contrast = abs(right_outer_level - right_inner_level)
    supported_contrast = min(
        left_region_contrast,
        right_region_contrast,
        left_edge_contrast,
        right_edge_contrast,
    )
    if supported_contrast < max(1.0, float(minimum_delta) * 0.5):
        return None

    local_inconsistency = abs(left_inner_level - inside_level) + abs(right_inner_level - inside_level)
    inside_mad = _median(np.abs(inside.astype(np.float64, copy=False) - inside_level))
    edge_strength = min(float(first.strength), float(second.strength))
    distance = float(second.position - first.position)
    balance_penalty = abs(left_region_contrast - right_region_contrast) * 0.15

    score = (
        distance * 3.0
        + supported_contrast
        + edge_strength * 0.15
        - local_inconsistency * 2.0
        - inside_mad * 0.15
        - balance_penalty
    )
    return score, distance, supported_contrast, edge_strength


def _best_stable_region_pair(
    scan_result: EdgeScanResult,
    scan_index: int,
    first_candidates: Sequence[RawEdgeCandidate],
    second_candidates: Sequence[RawEdgeCandidate],
) -> Optional[PairCandidate]:
    profile = _scanline_profile(scan_result, scan_index)
    if profile is None:
        return None

    best_key: Optional[tuple[float, float, float, float]] = None
    best_pair: Optional[PairCandidate] = None
    for first in first_candidates:
        for second in second_candidates:
            if second.position <= first.position:
                continue
            key = _stable_region_score(profile, first, second, scan_result.minimum_grayscale_delta)
            if key is None:
                continue
            if best_key is None or key > best_key:
                best_key = key
                best_pair = PairCandidate(
                    scan_axis=scan_result.scan_axis,
                    scan_index=int(scan_index),
                    local_scan_index=int(first.local_scan_index),
                    first=first,
                    second=second,
                )
    return best_pair


def select_stable_region_pairs_per_scanline(
    scan_result: EdgeScanResult,
    first_direction: Optional[str] = None,
    second_direction: Optional[str] = None,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
) -> List[PairCandidate]:
    midpoint = float(_scan_profile_length(scan_result) - 1) / 2.0
    selected_pairs: List[PairCandidate] = []
    for scan_index, candidates in _group_by_scanline(scan_result.raw_edge_candidates).items():
        first_candidates = _side_pool(candidates, midpoint, "first")
        second_candidates = _side_pool(candidates, midpoint, "second")
        pair = _best_stable_region_pair(scan_result, scan_index, first_candidates, second_candidates)
        if pair is not None:
            selected_pairs.append(pair)

    if not selected_pairs and not scan_result.profiles_by_scanline:
        return select_first_valid_boundary_pairs_per_scanline(scan_result, first_direction, second_direction)

    max_gap, smooth_window = _postprocess_defaults(scan_result.scan_axis)
    first = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, [pair.first for pair in selected_pairs], "first", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    second = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(scan_result, [pair.second for pair in selected_pairs], "second", max_jump_px),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    return _pairs_from_boundary_candidates(scan_result, first, second)


def select_first_valid_boundary_pairs_per_scanline(
    scan_result: EdgeScanResult,
    first_direction: Optional[str] = None,
    second_direction: Optional[str] = None,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
) -> List[PairCandidate]:
    max_gap, smooth_window = _postprocess_defaults(scan_result.scan_axis)
    first = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(
            scan_result,
            select_first_valid_boundary_candidates(scan_result, "first", first_direction),
            "first",
            max_jump_px,
        ),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    second = postprocess_boundary_candidates(
        scan_result,
        _repair_boundary_jumps(
            scan_result,
            select_first_valid_boundary_candidates(scan_result, "second", second_direction),
            "second",
            max_jump_px,
        ),
        max_gap=max_gap,
        smooth_window=smooth_window,
    )
    return _pairs_from_boundary_candidates(scan_result, first, second)


def select_refined_side_pair_per_scanline(scan_result: EdgeScanResult) -> List[PairCandidate]:
    return select_stable_region_pairs_per_scanline(scan_result)


def select_boundary_curve_candidates(
    scan_result: EdgeScanResult,
    side: str,
    direction: Optional[str] = None,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    prefer_sign: bool = False,
) -> List[RawEdgeCandidate]:
    max_gap, smooth_window = _postprocess_defaults(scan_result.scan_axis)
    selected = select_first_valid_boundary_candidates(scan_result, side, direction, prefer_sign=prefer_sign)
    repaired = _repair_boundary_jumps(scan_result, selected, side, max_jump_px)
    return postprocess_boundary_candidates(scan_result, repaired, max_gap=max_gap, smooth_window=smooth_window)
