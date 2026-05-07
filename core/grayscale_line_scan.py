from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from fib_sem_measurement_tool.models.result import EdgeScanResult, PairCandidate, RawEdgeCandidate
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _minimum_delta(settings: MeasurementSettings) -> float:
    return max(0.0, float(getattr(settings, "minimum_grayscale_delta", 30.0)))


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


def detect_profile_candidates(
    profile: Sequence[float],
    settings: MeasurementSettings,
    scan_axis: str,
    scan_index: int,
    local_scan_index: int,
    roi_origin: tuple[int, int],
) -> List[RawEdgeCandidate]:
    data = np.asarray(profile, dtype=np.float32).reshape(-1)
    if data.size < 2:
        return []

    diff = data[1:] - data[:-1]
    strength = np.abs(diff)
    min_delta = _minimum_delta(settings)

    candidates: List[RawEdgeCandidate] = []
    for i, signed_delta in enumerate(diff):
        delta = float(signed_delta)
        if delta == 0.0:
            continue
        if float(strength[i]) < min_delta:
            continue
        position = float(i) + 0.5
        image_x, image_y = _candidate_image_position(scan_axis, scan_index, position, roi_origin)
        candidates.append(
            RawEdgeCandidate(
                scan_axis=scan_axis,
                scan_index=int(scan_index),
                local_scan_index=int(local_scan_index),
                position=position,
                image_x=image_x,
                image_y=image_y,
                strength=float(strength[i]),
                signed_delta=delta,
                sign=1 if delta > 0.0 else -1,
                grayscale_before=float(data[i]),
                grayscale_after=float(data[i + 1]),
            )
        )
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
) -> Optional[RawEdgeCandidate]:
    if not candidates:
        return None
    if direction in {"left_to_center", "center_to_right", "top_to_bottom"}:
        return min(candidates, key=lambda candidate: candidate.position)
    if direction in {"center_to_left", "right_to_center", "bottom_to_top"}:
        return max(candidates, key=lambda candidate: candidate.position)
    raise ValueError(f"Unsupported edge scan direction: {direction}")


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
) -> List[RawEdgeCandidate]:
    midpoint = float(_scan_profile_length(scan_result) - 1) / 2.0
    scan_direction = direction or _default_direction(scan_result.scan_axis, side)
    selected: List[RawEdgeCandidate] = []
    for _scan_index, candidates in _group_by_scanline(scan_result.raw_edge_candidates).items():
        side_candidates = _side_pool(candidates, midpoint, side)
        candidate = scan_edge_in_direction(side_candidates, scan_direction)
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

    first = filter_boundary_candidates_by_continuity([pair.first for pair in selected_pairs])
    second = filter_boundary_candidates_by_continuity([pair.second for pair in selected_pairs])
    return _pairs_from_boundary_candidates(scan_result, first, second)


def select_first_valid_boundary_pairs_per_scanline(
    scan_result: EdgeScanResult,
    first_direction: Optional[str] = None,
    second_direction: Optional[str] = None,
) -> List[PairCandidate]:
    first = filter_boundary_candidates_by_continuity(
        select_first_valid_boundary_candidates(scan_result, "first", first_direction)
    )
    second = filter_boundary_candidates_by_continuity(
        select_first_valid_boundary_candidates(scan_result, "second", second_direction)
    )
    return _pairs_from_boundary_candidates(scan_result, first, second)


def select_refined_side_pair_per_scanline(scan_result: EdgeScanResult) -> List[PairCandidate]:
    return select_stable_region_pairs_per_scanline(scan_result)
