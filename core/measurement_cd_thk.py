from __future__ import annotations

from typing import List, Sequence

import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    scan_pair_candidates,
    scan_raw_edge_candidates,
    select_density_layer_pairs_per_scanline,
    select_first_valid_boundary_pairs_per_scanline,
    select_projection_layer_pairs_per_scanline,
    select_stable_region_pairs_per_scanline,
)
from fib_sem_measurement_tool.models.result import DistanceResult, MeasurementResult, PairCandidate
from fib_sem_measurement_tool.models.settings import MeasurementSettings


MIN_COVERAGE_FOR_OK = 0.45
THK_CD_MARGIN_RATIO = 0.075
THK_CD_MARGIN_MIN_PX = 3.0
THK_CD_MARGIN_MAX_PX = 5.0


def _status_from_candidates(selected_count: int, coverage: float) -> str:
    if selected_count <= 0:
        return "Fail"
    if coverage >= MIN_COVERAGE_FOR_OK:
        return "OK"
    return "Check"


def _selected_value(values: np.ndarray, method: str) -> float:
    if method == "max":
        return float(np.max(values))
    if method == "min":
        return float(np.min(values))
    return float(np.mean(values))


def _boundary_pairs(orientation: str, selected_pairs: Sequence[PairCandidate]) -> List[tuple[float, float, float]]:
    if orientation == "horizontal":
        return [
            (float(pair.scan_index), float(pair.first.image_x), float(pair.second.image_x))
            for pair in selected_pairs
        ]
    return [
        (float(pair.scan_index), float(pair.first.image_y), float(pair.second.image_y))
        for pair in selected_pairs
    ]


def _nearest_selected_pair(selected_pairs: Sequence[PairCandidate], selected_px: float | None) -> PairCandidate | None:
    if not selected_pairs or selected_px is None:
        return None
    return min(selected_pairs, key=lambda pair: abs(pair.distance - selected_px))


def _boundary_coordinate(orientation: str, pair: PairCandidate, boundary: str) -> float:
    candidate = pair.first if boundary == "first" else pair.second
    return float(candidate.image_x if orientation == "horizontal" else candidate.image_y)


def _local_boundary_angles_deg(
    selected_pairs: Sequence[PairCandidate],
    orientation: str,
    boundary: str,
    window: int,
) -> dict[int, float]:
    if len(selected_pairs) < 3:
        return {int(pair.scan_index): 0.0 for pair in selected_pairs}

    pairs = sorted(selected_pairs, key=lambda pair: int(pair.scan_index))
    scans = np.asarray([float(pair.scan_index) for pair in pairs], dtype=np.float64)
    coords = np.asarray([_boundary_coordinate(orientation, pair, boundary) for pair in pairs], dtype=np.float64)
    half = max(1, int(window) // 2)
    angles: dict[int, float] = {}

    for index, pair in enumerate(pairs):
        start = max(0, index - half)
        end = min(len(pairs), index + half + 1)
        if end - start < 3:
            start = max(0, min(start, len(pairs) - 3))
            end = min(len(pairs), max(end, start + 3))
        local_scans = scans[start:end]
        local_coords = coords[start:end]
        if len(np.unique(local_scans)) < 2:
            angle = 0.0
        else:
            slope, _intercept = np.polyfit(local_scans, local_coords, 1)
            angle = abs(float(np.degrees(np.arctan(float(slope)))))
        angles[int(pair.scan_index)] = angle
    return angles


def _filter_pairs_by_boundary_angle(
    selected_pairs: Sequence[PairCandidate],
    orientation: str,
    max_angle_deg: float,
    window: int = 9,
) -> tuple[list[PairCandidate], int]:
    if max_angle_deg <= 0.0 or max_angle_deg >= 89.9 or len(selected_pairs) < 3:
        return list(selected_pairs), 0

    first_angles = _local_boundary_angles_deg(selected_pairs, orientation, "first", window)
    second_angles = _local_boundary_angles_deg(selected_pairs, orientation, "second", window)
    filtered = [
        pair
        for pair in selected_pairs
        if first_angles.get(int(pair.scan_index), 0.0) <= max_angle_deg
        and second_angles.get(int(pair.scan_index), 0.0) <= max_angle_deg
    ]
    removed_count = len(selected_pairs) - len(filtered)
    return filtered, removed_count


def _edge_scan_directions(orientation: str, settings: MeasurementSettings) -> tuple[str, str]:
    edge_scan_mode = getattr(settings, "edge_scan_mode", "auto")
    if edge_scan_mode in {"outside_to_center", "center_to_outside"}:
        return edge_scan_mode, edge_scan_mode
    if orientation == "horizontal":
        return (
            getattr(settings, "cd_left_edge_direction", "left_to_center"),
            getattr(settings, "cd_right_edge_direction", "right_to_center"),
        )
    if orientation == "vertical":
        return (
            getattr(settings, "thk_top_edge_direction", "top_to_bottom"),
            getattr(settings, "thk_bottom_edge_direction", "bottom_to_top"),
        )
    raise ValueError(f"Unsupported scan orientation: {orientation}")


def derive_thk_roi_from_cd(roi: Sequence[int], cd_result: DistanceResult | None) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in roi]
    if cd_result is None or not cd_result.selected_pairs:
        return x1, y1, x2, y2

    left_edges = []
    right_edges = []
    for pair in cd_result.selected_pairs:
        if pair.second.image_x <= pair.first.image_x:
            continue
        left_edges.append(float(pair.first.image_x))
        right_edges.append(float(pair.second.image_x))
    if not left_edges or not right_edges:
        return x1, y1, x2, y2

    left_median = float(np.median(np.asarray(left_edges, dtype=np.float64)))
    right_median = float(np.median(np.asarray(right_edges, dtype=np.float64)))
    cd_width = right_median - left_median
    if cd_width <= 1.0:
        return x1, y1, x2, y2

    margin = max(THK_CD_MARGIN_MIN_PX, min(THK_CD_MARGIN_MAX_PX, cd_width * THK_CD_MARGIN_RATIO))
    thk_x_min = max(x1, int(np.ceil(left_median + margin)))
    thk_x_max = min(x2, int(np.floor(right_median - margin)))
    if thk_x_max <= thk_x_min:
        thk_x_min = max(x1, int(np.ceil(left_median)))
        thk_x_max = min(x2, int(np.floor(right_median)))
    if thk_x_max <= thk_x_min:
        return x1, y1, x2, y2
    return thk_x_min, y1, thk_x_max, y2


def _measure_distance(
    gray: np.ndarray,
    roi: Sequence[int],
    orientation: str,
    settings: MeasurementSettings,
) -> DistanceResult:
    scan = scan_raw_edge_candidates(gray, roi, orientation, settings)
    pair_candidates = scan_pair_candidates(scan)
    first_direction, second_direction = _edge_scan_directions(orientation, settings)
    max_jump_px = float(getattr(settings, "max_jump_px", 28.0))
    selected_pairs = []
    edge_scan_mode = getattr(settings, "edge_scan_mode", "auto")
    if edge_scan_mode in {"outside_to_center", "center_to_outside"}:
        selected_pairs = select_first_valid_boundary_pairs_per_scanline(
            scan,
            first_direction,
            second_direction,
            max_jump_px=max_jump_px,
        )
    elif orientation == "vertical":
        selected_pairs = select_projection_layer_pairs_per_scanline(scan, max_jump_px=max_jump_px)
        if not selected_pairs:
            selected_pairs = select_density_layer_pairs_per_scanline(scan, max_jump_px=max_jump_px)
    if not selected_pairs:
        selected_pairs = select_stable_region_pairs_per_scanline(
            scan,
            first_direction,
            second_direction,
            max_jump_px=max_jump_px,
        )

    angle_filtered_count = 0
    if bool(getattr(settings, "filter_cd_thk_by_boundary_angle", False)):
        selected_pairs, angle_filtered_count = _filter_pairs_by_boundary_angle(
            selected_pairs,
            orientation,
            max_angle_deg=float(getattr(settings, "max_cd_thk_boundary_angle_deg", 18.0)),
            window=int(getattr(settings, "cd_thk_boundary_angle_window", 9)),
        )

    result = DistanceResult(
        orientation=orientation,
        selected_method=settings.distance_method,
        total_count=scan.scanned_line_count,
        scanned_line_count=scan.scanned_line_count,
        valid_scanline_count=scan.valid_scanline_count,
        scanline_coverage=scan.scanline_coverage,
        raw_edge_density=scan.raw_edge_density,
        minimum_grayscale_delta=scan.minimum_grayscale_delta,
        raw_edge_candidates=list(scan.raw_edge_candidates),
        pair_candidates=list(pair_candidates),
        selected_pairs=list(selected_pairs),
        per_scanline_candidate_count=dict(scan.per_scanline_candidate_count),
        raw_edge_count=scan.raw_edge_count,
        pair_candidate_count=len(pair_candidates),
        selected_point_count=len(selected_pairs) * 2,
    )

    if not selected_pairs:
        if angle_filtered_count:
            result.warning_message = f"{orientation} boundary angle filter removed all selected scanlines"
        else:
            result.warning_message = f"{orientation} raw grayscale pair candidates not found"
        result.status = "Fail"
        return result

    values = np.asarray([pair.distance for pair in selected_pairs], dtype=np.float32)
    result.values_px = [float(value) for value in values]
    result.valid_count = int(values.size)
    selected_pair_coverage = float(result.valid_count / result.scanned_line_count) if result.scanned_line_count else 0.0
    result.mean_px = float(np.mean(values))
    result.max_px = float(np.max(values))
    result.min_px = float(np.min(values))
    result.median_px = float(np.median(values))
    result.std_px = float(np.std(values))
    result.selected_px = _selected_value(values, settings.distance_method)
    result.boundary_pairs = _boundary_pairs(orientation, selected_pairs)
    result.selected_pair = _nearest_selected_pair(selected_pairs, result.selected_px)
    result.confidence = float(selected_pair_coverage * 100.0)
    result.status = _status_from_candidates(result.valid_count, selected_pair_coverage)
    if angle_filtered_count:
        result.warning_message = f"{orientation} boundary angle filter removed {angle_filtered_count} scanlines"
    return result


def measure_horizontal_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    return _measure_distance(gray, roi, "horizontal", settings)


def measure_vertical_thk(
    gray: np.ndarray,
    roi: Sequence[int],
    settings: MeasurementSettings,
    cd_result: DistanceResult | None = None,
) -> DistanceResult:
    thk_roi = derive_thk_roi_from_cd(roi, cd_result)
    return _measure_distance(gray, thk_roi, "vertical", settings)


class LimitPeakBoundaryEngine:
    def __init__(self, settings: MeasurementSettings):
        self.settings = settings

    def measure(
        self,
        gray: np.ndarray,
        roi: Sequence[int],
        measure_direction: str | None = None,
    ) -> dict[str, DistanceResult]:
        direction = measure_direction or getattr(self.settings, "measure_direction", "both")
        if direction not in {"horizontal", "vertical", "both"}:
            raise ValueError(f"Unsupported measure direction: {direction}")

        results: dict[str, DistanceResult] = {}
        horizontal: DistanceResult | None = None
        if direction in {"horizontal", "both"}:
            horizontal = measure_horizontal_cd(gray, roi, self.settings)
            results["horizontal"] = horizontal
        if direction in {"vertical", "both"}:
            results["vertical"] = measure_vertical_thk(gray, roi, self.settings, horizontal)
        return results


class CDMeasurementEngine:
    def __init__(self, settings: MeasurementSettings):
        self.settings = settings
        self.boundary_engine = LimitPeakBoundaryEngine(settings)

    def measure(self, gray: np.ndarray, roi: Sequence[int], measure_direction: str | None = None) -> MeasurementResult:
        direction = measure_direction or getattr(self.settings, "measure_direction", "both")
        results = self.boundary_engine.measure(gray, roi, direction)
        horizontal = results.get("horizontal")
        vertical = results.get("vertical")
        warning_message = "; ".join(
            item.warning_message for item in (horizontal, vertical) if item is not None and item.warning_message
        )
        return MeasurementResult(
            measurement_type=f"distance_{direction}",
            horizontal_cd=horizontal,
            vertical_thk=vertical,
            warning_message=warning_message,
        )
