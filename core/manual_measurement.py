from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.models.result import (
    DistanceResult,
    CraterResult,
    MeasurementResult,
    MeasurementStatus,
    PairCandidate,
    RawEdgeCandidate,
    TaperSideResult,
    HoleCDResult,
)
from fib_sem_measurement_tool.models.settings import MeasurementSettings

Point = Tuple[float, float]


def required_manual_points(measurement_type: str) -> int:
    if measurement_type in {"distance_both", "hole_cd", "taper_double"}:
        return 4
    if measurement_type == "crater":
        return 3
    return 2


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
        constrained_p1 = (float(p1[0]), float(scan_index))
        constrained_p2 = (float(p2[0]), float(scan_index))
        first = _candidate(constrained_p1, scan_axis, scan_index, constrained_p1[0])
        second = _candidate(constrained_p2, scan_axis, scan_index, constrained_p2[0])
    else:
        value = abs(float(p2[1]) - float(p1[1]))
        scan_axis = "y"
        scan_index = int(round((float(p1[0]) + float(p2[0])) * 0.5))
        constrained_p1 = (float(scan_index), float(p1[1]))
        constrained_p2 = (float(scan_index), float(p2[1]))
        first = _candidate(constrained_p1, scan_axis, scan_index, constrained_p1[1])
        second = _candidate(constrained_p2, scan_axis, scan_index, constrained_p2[1])

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


def _selected_pair_points(result: DistanceResult) -> tuple[Point, Point]:
    pair = result.selected_pair or result.selected_pairs[0]
    return (pair.first.image_x, pair.first.image_y), (pair.second.image_x, pair.second.image_y)


def _manual_hole_result(horizontal_result: DistanceResult, vertical_result: DistanceResult, settings: MeasurementSettings) -> HoleCDResult:
    h1, h2 = _selected_pair_points(horizontal_result)
    v1, v2 = _selected_pair_points(vertical_result)
    horizontal = float(horizontal_result.selected_px or 0.0)
    vertical = float(vertical_result.selected_px or 0.0)
    cx = (float(h1[0]) + float(h2[0])) * 0.5
    cy = (float(v1[1]) + float(v2[1])) * 0.5
    rx = max(horizontal * 0.5, 1.0)
    ry = max(vertical * 0.5, 1.0)
    contour = [
        (cx + rx * math.cos(theta), cy + ry * math.sin(theta))
        for theta in np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    ]
    area = math.pi * rx * ry
    perimeter = math.pi * (3 * (rx + ry) - math.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    return HoleCDResult(
        target=getattr(settings, "hole_target", "inner"),
        horizontal_px=horizontal,
        vertical_px=vertical,
        min_feret_px=min(horizontal, vertical),
        max_feret_px=max(horizontal, vertical),
        equivalent_diameter_px=math.sqrt(4.0 * area / math.pi),
        area_px=area,
        perimeter_px=perimeter,
        coverage=1.0,
        mean_radius=(rx + ry) * 0.5,
        radius_std=abs(rx - ry) * 0.5,
        mean_strength=1.0,
        smoothness=0.0,
        continuity=1.0,
        confidence=100.0,
        status=MeasurementStatus.OK.value,
        contour_points=contour,
        center=(cx, cy),
        ellipse_major_px=max(horizontal, vertical),
        ellipse_minor_px=min(horizontal, vertical),
        ellipse_angle_deg=0.0,
        ellipse_fit_error=0.0,
        ellipse_aspect_ratio=(max(horizontal, vertical) / max(min(horizontal, vertical), 1e-6)),
    )


def _manual_crater_result(horizontal_result: DistanceResult, vertical_result: DistanceResult, settings: MeasurementSettings) -> CraterResult:
    h1, h2 = _selected_pair_points(horizontal_result)
    v1, v2 = _selected_pair_points(vertical_result)
    left, right = (h1, h2) if h1[0] <= h2[0] else (h2, h1)
    top, bottom = (v1, v2) if v1[1] <= v2[1] else (v2, v1)
    cd_px = float(horizontal_result.selected_px or 0.0)
    thk_px = float(vertical_result.selected_px or 0.0)
    px_to_real = float(settings.calibration.px_to_real or 1.0)
    center_x = float(top[0])
    baseline_y = float(left[1])
    side_height = max(thk_px, 1.0)
    left_boundary = [(float(left[0]), baseline_y), (float(left[0]), baseline_y - side_height * 0.5)]
    right_boundary = [(float(right[0]), baseline_y), (float(right[0]), baseline_y - side_height * 0.5)]
    top_profile = [(float(left[0]), float(top[1])), (center_x, float(top[1])), (float(right[0]), float(top[1]))]
    return CraterResult(
        cd_px=cd_px,
        cd=cd_px * px_to_real,
        thk_px=thk_px,
        thk=thk_px * px_to_real,
        thk_mean_px=thk_px,
        thk_max_px=thk_px,
        thk_min_px=thk_px,
        thk_median_px=thk_px,
        left_foot_x=float(left[0]),
        left_foot_y=baseline_y,
        right_foot_x=float(right[0]),
        right_foot_y=baseline_y,
        baseline_y_left=baseline_y,
        baseline_y_right=baseline_y,
        baseline_slope=0.0,
        baseline_intercept=baseline_y,
        baseline_confidence=100.0,
        baseline_status=MeasurementStatus.OK.value,
        center_x=center_x,
        top_y_at_center=float(top[1]),
        baseline_y_at_center=baseline_y,
        top_profile_point_count=len(top_profile),
        top_profile_valid_count=len(top_profile),
        top_profile_coverage=1.0,
        top_profile_smoothness=0.0,
        top_profile_confidence=100.0,
        top_profile_status=MeasurementStatus.OK.value,
        foot_confidence=100.0,
        foot_status=MeasurementStatus.OK.value,
        cd_status=MeasurementStatus.OK.value,
        cd_confidence=100.0,
        thk_status=MeasurementStatus.OK.value,
        thk_confidence=100.0,
        confidence=100.0,
        overall_confidence=100.0,
        status=MeasurementStatus.OK.value,
        top_profile_points=top_profile,
        left_boundary_points=left_boundary,
        right_boundary_points=right_boundary,
        baseline_line=(float(left[0]), baseline_y, float(right[0]), baseline_y),
        cd_line=(float(left[0]), baseline_y, float(right[0]), baseline_y),
        thk_line=(center_x, float(top[1]), center_x, float(bottom[1])),
    )


def _manual_crater_from_three_points(points: Sequence[Point], settings: MeasurementSettings) -> tuple[DistanceResult, DistanceResult, CraterResult]:
    left_point, right_point, top_point = _ordered_points(points[:3])
    horizontal_result = _distance_result(left_point, right_point, "horizontal", settings.distance_method)
    h1, h2 = _selected_pair_points(horizontal_result)
    baseline_y = float(h1[1])
    top_x = float(top_point[0])
    vertical_result = _distance_result((top_x, float(top_point[1])), (top_x, baseline_y), "vertical", settings.distance_method)
    crater = _manual_crater_result(horizontal_result, vertical_result, settings)
    crater.center_x = top_x
    crater.top_y_at_center = float(top_point[1])
    crater.baseline_y_at_center = baseline_y
    crater.thk_line = (top_x, float(top_point[1]), top_x, baseline_y)
    if crater.top_profile_points:
        left, right = (h1, h2) if h1[0] <= h2[0] else (h2, h1)
        crater.top_profile_points = [(float(left[0]), float(top_point[1])), (top_x, float(top_point[1])), (float(right[0]), float(top_point[1]))]
    return horizontal_result, vertical_result, crater


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
        result.vertical_thk = _distance_result(ordered[2], ordered[3], "vertical", settings.distance_method)
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
    elif measurement_type == "hole_cd":
        result.horizontal_cd = _distance_result(ordered[0], ordered[1], "horizontal", settings.distance_method)
        result.vertical_thk = _distance_result(ordered[2], ordered[3], "vertical", settings.distance_method)
        result.hole_cd = _manual_hole_result(result.horizontal_cd, result.vertical_thk, settings)
    elif measurement_type == "crater":
        result.horizontal_cd, result.vertical_thk, result.crater = _manual_crater_from_three_points(ordered, settings)
    else:
        result.status = MeasurementStatus.FAIL.value
        result.overall_confidence = 0.0
        result.warning_message = "Unsupported manual measurement type"

    from fib_sem_measurement_tool.core.measurement_runner import calculate_overall_coverage

    calculate_overall_coverage(result)
    result.measurement_source = "manual"
    return result
