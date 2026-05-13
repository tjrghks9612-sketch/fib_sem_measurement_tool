from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.measurement_runner import calculate_overall_coverage
from fib_sem_measurement_tool.models.result import (
    DistanceResult,
    MeasurementResult,
    MeasurementStatus,
    PairCandidate,
    RawEdgeCandidate,
    TaperSideResult,
)
from fib_sem_measurement_tool.models.settings import MeasurementSettings

Point = Tuple[float, float]


def required_manual_points(measurement_type: str) -> int:
    return 4 if measurement_type == "taper_double" else 2


def _candidate(point: Point, scan_axis: str, scan_index: int, position: float) -> RawEdgeCandidate:
    x, y = point
    return RawEdgeCandidate(
        scan_axis=scan_axis,
        scan_index=int(scan_index),
        local_scan_index=int(scan_index),
        position=float(position),
        image_x=float(x),
        image_y=float(y),
        strength=1.0,
        signed_delta=1.0,
        sign=1,
        grayscale_before=0.0,
        grayscale_after=0.0,
    )


def _distance_result(p1: Point, p2: Point, orientation: str, method: str) -> DistanceResult:
    if orientation == "horizontal":
        value = abs(float(p2[0]) - float(p1[0]))
        scan_axis = "x"
        scan_index = int(round((float(p1[1]) + float(p2[1])) * 0.5))
        first = _candidate(p1, scan_axis, scan_index, p1[0])
        second = _candidate(p2, scan_axis, scan_index, p2[0])
    else:
        value = abs(float(p2[1]) - float(p1[1]))
        scan_axis = "y"
        scan_index = int(round((float(p1[0]) + float(p2[0])) * 0.5))
        first = _candidate(p1, scan_axis, scan_index, p1[1])
        second = _candidate(p2, scan_axis, scan_index, p2[1])

    if second.position < first.position:
        first, second = second, first
    pair = PairCandidate(scan_axis=scan_axis, scan_index=scan_index, local_scan_index=scan_index, first=first, second=second)
    return DistanceResult(
        orientation=orientation,
        mean_px=value,
        max_px=value,
        min_px=value,
        median_px=value,
        std_px=0.0,
        selected_px=value,
        selected_method=method,
        valid_count=1,
        total_count=1,
        confidence=100.0,
        status=MeasurementStatus.OK.value,
        boundary_pairs=[(float(first.position), float(second.position), value)],
        values_px=[value],
        raw_edge_candidates=[first, second],
        pair_candidates=[pair],
        selected_pairs=[pair],
        selected_pair=pair,
        raw_edge_count=2,
        pair_candidate_count=1,
        scanned_line_count=1,
        valid_scanline_count=1,
        scanline_coverage=1.0,
        selected_point_count=2,
        raw_edge_density=2.0,
    )


def _line_angles(p1: Point, p2: Point) -> tuple[float, float]:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    angle_horizontal = abs(math.degrees(math.atan2(dy, dx)))
    if angle_horizontal > 90.0:
        angle_horizontal = 180.0 - angle_horizontal
    angle_vertical = abs(math.degrees(math.atan2(dx, dy)))
    if angle_vertical > 90.0:
        angle_vertical = 180.0 - angle_vertical
    return float(angle_horizontal), float(angle_vertical)


def _taper_result(p1: Point, p2: Point, side: str) -> TaperSideResult:
    angle_horizontal, angle_vertical = _line_angles(p1, p2)
    candidates = [
        _candidate(p1, "manual_taper", int(round(p1[1])), p1[0]),
        _candidate(p2, "manual_taper", int(round(p2[1])), p2[0]),
    ]
    return TaperSideResult(
        side=side,
        angle_horizontal=angle_horizontal,
        angle_vertical=angle_vertical,
        fit_r2=1.0,
        fit_error=0.0,
        valid_point_count=2,
        inlier_count=2,
        confidence=100.0,
        status=MeasurementStatus.OK.value,
        points=[(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))],
        fit_line=(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])),
        raw_edge_candidates=candidates,
        selected_boundary_candidates=candidates,
        raw_edge_count=2,
        scanned_line_count=2,
        valid_scanline_count=2,
        scanline_coverage=1.0,
        selected_point_count=2,
        fit_point_count=2,
        raw_edge_density=1.0,
    )


def _ordered_points(points: Sequence[Point]) -> list[Point]:
    return [(float(x), float(y)) for x, y in points]


def make_manual_measurement(points: Sequence[Point], settings: MeasurementSettings) -> MeasurementResult:
    measurement_type = settings.measurement_type
    required = required_manual_points(measurement_type)
    if len(points) < required:
        return MeasurementResult(
            measurement_type=measurement_type,
            overall_confidence=0.0,
            status=MeasurementStatus.FAIL.value,
            warning_message=f"Manual measurement requires {required} points",
            measurement_source="manual",
        )

    ordered = _ordered_points(points[:required])
    result = MeasurementResult(
        measurement_type=measurement_type,
        overall_confidence=100.0,
        status=MeasurementStatus.OK.value,
        measurement_source="manual",
    )
    if measurement_type == "distance_horizontal":
        result.horizontal_cd = _distance_result(ordered[0], ordered[1], "horizontal", settings.distance_method)
    elif measurement_type == "distance_vertical":
        result.vertical_thk = _distance_result(ordered[0], ordered[1], "vertical", settings.distance_method)
    elif measurement_type == "distance_both":
        result.horizontal_cd = _distance_result(ordered[0], ordered[1], "horizontal", settings.distance_method)
        result.vertical_thk = _distance_result(ordered[0], ordered[1], "vertical", settings.distance_method)
    elif measurement_type == "taper_single":
        side = settings.taper_side if settings.taper_side in {"left", "right"} else "left"
        taper = _taper_result(ordered[0], ordered[1], side)
        if side == "right":
            result.right_taper = taper
        else:
            result.left_taper = taper
    elif measurement_type == "taper_double":
        result.left_taper = _taper_result(ordered[0], ordered[1], "left")
        result.right_taper = _taper_result(ordered[2], ordered[3], "right")
        angles = [
            taper.angle_horizontal
            for taper in (result.left_taper, result.right_taper)
            if taper is not None and taper.angle_horizontal is not None
        ]
        if angles:
            result.avg_taper_angle = float(np.mean(angles))
        if len(angles) == 2:
            result.taper_angle_diff = float(abs(angles[0] - angles[1]))
    else:
        result.status = MeasurementStatus.FAIL.value
        result.overall_confidence = 0.0
        result.warning_message = "Unsupported manual measurement type"

    calculate_overall_coverage(result)
    result.measurement_source = "manual"
    return result
