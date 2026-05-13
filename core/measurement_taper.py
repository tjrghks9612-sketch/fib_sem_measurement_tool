from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    fill_small_gaps_nan,
    moving_average_nan,
    refine_boundary_candidates_by_line,
)
from fib_sem_measurement_tool.models.result import EdgeScanResult, MeasurementResult, RawEdgeCandidate, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


MIN_TAPER_POINTS = 2
MIN_COVERAGE_FOR_OK = 0.45
TAPER_TARGET_WINDOW_PCT = 18.0
TAPER_MIN_LOCAL_POINTS = 5
TAPER_RESIDUAL_LIMIT_PX = 18.0
TAPER_FIT_PREFILTER_RESIDUAL_PX = 6.0
TAPER_GAP_INTERPOLATION_LIMIT = 6
TAPER_SMOOTH_WINDOW = 7
TAPER_FINAL_SMOOTH_KERNEL = 4


def _status_from_points(point_count: int, coverage: float) -> str:
    if point_count < MIN_TAPER_POINTS:
        return "Fail"
    if coverage >= MIN_COVERAGE_FOR_OK:
        return "OK"
    return "Check"


def _result_with_scan_metadata(side: str, scan) -> TaperSideResult:
    return TaperSideResult(
        side=side,
        raw_edge_candidates=list(scan.raw_edge_candidates),
        raw_edge_count=scan.raw_edge_count,
        scanned_line_count=scan.scanned_line_count,
        valid_scanline_count=scan.valid_scanline_count,
        scanline_coverage=scan.scanline_coverage,
        raw_edge_density=scan.raw_edge_density,
        per_scanline_candidate_count=dict(scan.per_scanline_candidate_count),
        minimum_grayscale_delta=scan.minimum_grayscale_delta,
    )


def _gradient_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _transition_values(profile: np.ndarray, position: float) -> tuple[float, float]:
    if profile.size == 0:
        return 0.0, 0.0
    center = int(round(float(position)))
    left = max(0, center - 1)
    right = min(profile.size - 1, center + 1)
    return float(profile[left]), float(profile[right])


class LimitPeakBoundaryEngine:
    """Taper-only absolute Sobel peak boundary detector."""

    def __init__(self, settings: MeasurementSettings):
        self.settings = settings
        self.limit = max(0.0, float(getattr(settings, "minimum_grayscale_delta", 55.0)))
        self.max_jump_px = max(0.0, float(getattr(settings, "max_jump_px", 28.0)))
        self.scan_mode = getattr(settings, "edge_scan_mode", "auto")
        if self.scan_mode == "auto":
            self.scan_mode = "center_to_outside"

    def scan(self, gray: np.ndarray, roi: Sequence[int], side: str) -> tuple[EdgeScanResult, list[RawEdgeCandidate]]:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported taper side: {side}")

        x1, y1, x2, y2 = [int(v) for v in roi]
        crop = np.asarray(gray[y1 : y2 + 1, x1 : x2 + 1])
        height = int(crop.shape[0]) if crop.ndim >= 2 else 0
        width = int(crop.shape[1]) if crop.ndim >= 2 else 0
        scan_result = EdgeScanResult(
            scan_axis="horizontal",
            roi=(x1, y1, x2, y2),
            scanned_line_count=height,
            minimum_grayscale_delta=self.limit,
        )
        if height <= 0 or width < 3:
            return scan_result, []

        if crop.ndim == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop
        crop_f = np.asarray(crop_gray, dtype=np.float32)
        blurred = cv2.GaussianBlur(crop_f, (5, 5), 0)
        gradient = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        strength = np.abs(gradient)
        scan_result.gradient_projection = np.mean(strength, axis=0).astype(np.float64).tolist()
        scan_result.profiles_by_scanline = {
            int(y1 + local_y): crop_f[local_y, :].astype(np.float64).tolist()
            for local_y in range(height)
        }

        split = width // 2
        raw_by_row: list[list[RawEdgeCandidate]] = []
        raw_all: list[RawEdgeCandidate] = []
        for local_y in range(height):
            row_candidates = self._row_candidates(
                gradient[local_y, :],
                strength[local_y, :],
                crop_f[local_y, :],
                roi_origin=(x1, y1),
                local_y=local_y,
                side=side,
                split=split,
            )
            raw_by_row.append(row_candidates)
            raw_all.extend(row_candidates)
            scan_result.per_scanline_candidate_count[int(y1 + local_y)] = len(row_candidates)

        scan_result.raw_edge_candidates = raw_all
        selected = self._select_continuous_boundary(raw_by_row, x1, y1, side)
        return scan_result, selected

    def _row_candidates(
        self,
        signed_gradient: np.ndarray,
        strength: np.ndarray,
        profile: np.ndarray,
        roi_origin: tuple[int, int],
        local_y: int,
        side: str,
        split: int,
    ) -> list[RawEdgeCandidate]:
        x1, y1 = roi_origin
        width = int(strength.size)
        if side == "left":
            start, end = 0, max(0, split)
        else:
            start, end = max(0, split), width - 1

        candidates: list[RawEdgeCandidate] = []
        for idx in range(max(1, start), min(width - 2, end) + 1):
            value = float(strength[idx])
            if value < self.limit:
                continue
            if value < float(strength[idx - 1]) or value < float(strength[idx + 1]):
                continue
            before, after = _transition_values(profile, float(idx))
            signed = float(signed_gradient[idx])
            candidates.append(
                RawEdgeCandidate(
                    scan_axis="horizontal",
                    scan_index=int(y1 + local_y),
                    local_scan_index=int(local_y),
                    position=float(idx),
                    image_x=float(x1 + idx),
                    image_y=float(y1 + local_y),
                    strength=value,
                    signed_delta=signed,
                    sign=_gradient_sign(signed),
                    grayscale_before=before,
                    grayscale_after=after,
                )
            )
        return candidates

    def _direction_rank(self, candidate: RawEdgeCandidate, side: str) -> float:
        if self.scan_mode == "outside_to_center":
            return -candidate.position if side == "left" else candidate.position
        return candidate.position if side == "left" else -candidate.position

    def _primary_candidate(self, candidates: Sequence[RawEdgeCandidate], side: str) -> RawEdgeCandidate | None:
        if not candidates:
            return None
        return max(candidates, key=lambda item: (float(item.strength), self._direction_rank(item, side)))

    def _select_continuous_boundary(
        self,
        raw_by_row: Sequence[Sequence[RawEdgeCandidate]],
        x1: int,
        y1: int,
        side: str,
    ) -> list[RawEdgeCandidate]:
        values = np.full(len(raw_by_row), np.nan, dtype=np.float64)
        templates: list[RawEdgeCandidate | None] = [None] * len(raw_by_row)
        previous: float | None = None

        for row_index, candidates in enumerate(raw_by_row):
            row = list(candidates)
            if not row:
                continue
            primary = self._primary_candidate(row, side)
            if primary is None:
                continue
            chosen = primary
            if previous is not None and abs(chosen.position - previous) > self.max_jump_px:
                alternatives = [
                    candidate
                    for candidate in row
                    if candidate is not chosen and abs(candidate.position - previous) <= self.max_jump_px
                ]
                if not alternatives:
                    continue
                chosen = min(alternatives, key=lambda item: abs(item.position - previous))
            values[row_index] = float(chosen.position)
            templates[row_index] = chosen
            previous = float(chosen.position)

        processed = fill_small_gaps_nan(values, TAPER_GAP_INTERPOLATION_LIMIT)
        processed = moving_average_nan(processed, TAPER_SMOOTH_WINDOW)
        processed = fill_small_gaps_nan(processed, TAPER_GAP_INTERPOLATION_LIMIT)
        processed = moving_average_nan(processed, TAPER_FINAL_SMOOTH_KERNEL)
        return [
            self._candidate_from_processed(row_index, position, templates[row_index], x1, y1)
            for row_index, position in enumerate(processed)
            if np.isfinite(position)
        ]

    def _candidate_from_processed(
        self,
        local_y: int,
        position: float,
        template: RawEdgeCandidate | None,
        x1: int,
        y1: int,
    ) -> RawEdgeCandidate:
        return RawEdgeCandidate(
            scan_axis="horizontal",
            scan_index=int(y1 + local_y),
            local_scan_index=int(local_y),
            position=float(position),
            image_x=float(x1 + position),
            image_y=float(y1 + local_y),
            strength=float(template.strength) if template is not None else 0.0,
            signed_delta=float(template.signed_delta) if template is not None else 0.0,
            sign=int(template.sign) if template is not None else 0,
            grayscale_before=float(template.grayscale_before) if template is not None else 0.0,
            grayscale_after=float(template.grayscale_after) if template is not None else 0.0,
        )


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _target_y_for_side(roi: Sequence[int], side: str, settings: MeasurementSettings) -> float:
    _x1, y1, _x2, y2 = [int(v) for v in roi]
    base_pct = float(getattr(settings, "base_height_pct", 30.0))
    offset_pct = float(
        getattr(settings, "right_offset_pct", 0.0)
        if side == "right"
        else getattr(settings, "left_offset_pct", 0.0)
    )
    target_pct = _clamp_pct(base_pct + offset_pct)
    return float(y2) - (float(y2 - y1) * target_pct / 100.0)


def _target_y_from_boundary(points: np.ndarray, side: str, settings: MeasurementSettings) -> float:
    finite = points[np.isfinite(points[:, 1])]
    if finite.size == 0:
        return 0.0
    top_y = float(np.min(finite[:, 1]))
    bottom_y = float(np.max(finite[:, 1]))
    base_pct = float(getattr(settings, "base_height_pct", 30.0))
    offset_pct = float(
        getattr(settings, "right_offset_pct", 0.0)
        if side == "right"
        else getattr(settings, "left_offset_pct", 0.0)
    )
    target_pct = _clamp_pct(base_pct + offset_pct)
    return bottom_y - (bottom_y - top_y) * (target_pct / 100.0)


def _fit_candidates_near_target(
    candidates: Sequence[RawEdgeCandidate],
    target_y: float,
    roi: Sequence[int],
) -> list[RawEdgeCandidate]:
    if len(candidates) <= MIN_TAPER_POINTS:
        return list(candidates)

    points = np.asarray([(candidate.image_x, candidate.image_y) for candidate in candidates], dtype=np.float64)
    _x1, y1, _x2, y2 = [int(v) for v in roi]
    height = max(1.0, float(y2 - y1 + 1))
    radius = max(4.0, height * TAPER_TARGET_WINDOW_PCT / 200.0)
    distances = np.abs(points[:, 1] - float(target_y))
    indices = np.flatnonzero(distances <= radius)
    if indices.size < MIN_TAPER_POINTS:
        needed = min(max(TAPER_MIN_LOCAL_POINTS, MIN_TAPER_POINTS), len(candidates))
        indices = np.argsort(distances)[:needed]
    return [candidates[int(index)] for index in np.sort(indices)]


def _prefilter_taper_fit_candidates(candidates: Sequence[RawEdgeCandidate]) -> list[RawEdgeCandidate]:
    selected = list(candidates)
    if len(selected) < TAPER_MIN_LOCAL_POINTS:
        return selected

    refined = refine_boundary_candidates_by_line(
        selected,
        residual_limit_px=TAPER_FIT_PREFILTER_RESIDUAL_PX,
        mad_multiplier=2.5,
        iterations=5,
    )
    if len(refined) < TAPER_MIN_LOCAL_POINTS:
        return selected
    if len(refined) < max(TAPER_MIN_LOCAL_POINTS, int(len(selected) * 0.35)):
        return selected
    return refined


def _line_angles(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    angle_horizontal = abs(math.degrees(math.atan2(dy, dx)))
    if angle_horizontal > 90.0:
        angle_horizontal = 180.0 - angle_horizontal
    angle_vertical = abs(math.degrees(math.atan2(dx, dy)))
    if angle_vertical > 90.0:
        angle_vertical = 180.0 - angle_vertical
    return float(angle_horizontal), float(angle_vertical)


def _fit_selected_boundary(
    result: TaperSideResult,
    selected: Sequence[RawEdgeCandidate],
    roi: Sequence[int],
    settings: MeasurementSettings,
) -> TaperSideResult:
    all_selected = list(selected)
    result.selected_boundary_candidates = list(all_selected)
    result.points = [(float(candidate.image_x), float(candidate.image_y)) for candidate in all_selected]
    result.selected_point_count = len(all_selected)

    if len(all_selected) < MIN_TAPER_POINTS:
        result.fit_point_count = len(all_selected)
        result.valid_point_count = len(all_selected)
        result.inlier_count = len(all_selected)
        result.confidence = 0.0
        result.status = _status_from_points(result.valid_point_count, 0.0)
        result.warning_message = f"{result.side} raw grayscale edge points not found"
        return result

    all_pts_for_target = np.asarray([(candidate.image_x, candidate.image_y) for candidate in all_selected], dtype=np.float64)
    target_y = _target_y_from_boundary(all_pts_for_target, result.side, settings)
    fit_source_candidates = _fit_candidates_near_target(all_selected, target_y, roi)
    fit_candidates = _prefilter_taper_fit_candidates(fit_source_candidates)
    fit_pts = np.asarray([(candidate.image_x, candidate.image_y) for candidate in fit_candidates], dtype=np.float64)
    result.fit_point_count = int(fit_pts.shape[0])
    result.valid_point_count = int(fit_pts.shape[0])
    selected_coverage = float(len(all_selected) / result.scanned_line_count) if result.scanned_line_count else 0.0
    result.confidence = float(selected_coverage * 100.0)
    result.status = _status_from_points(result.valid_point_count, selected_coverage)

    if fit_pts.shape[0] < MIN_TAPER_POINTS:
        result.warning_message = f"{result.side} target taper fit points not found"
        return result

    xs, ys = fit_pts[:, 0], fit_pts[:, 1]
    m, b = np.polyfit(ys, xs, 1)
    pred = m * ys + b
    residuals = xs - pred
    err = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((xs - np.mean(xs)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-9 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    result.fit_r2 = float(r2)
    result.fit_error = err

    residual_limit = max(4.0, float(getattr(settings, "taper_residual_limit_px", TAPER_RESIDUAL_LIMIT_PX)))
    all_pts = np.asarray([(candidate.image_x, candidate.image_y) for candidate in all_selected], dtype=np.float64)
    all_residuals = np.abs(all_pts[:, 0] - (m * all_pts[:, 1] + b))
    inliers = [
        candidate
        for candidate, residual in zip(all_selected, all_residuals)
        if float(residual) <= residual_limit
    ]
    if len(inliers) < MIN_TAPER_POINTS:
        inliers = list(fit_candidates)
    result.selected_boundary_candidates = list(inliers)
    result.points = [(float(candidate.image_x), float(candidate.image_y)) for candidate in inliers]
    result.selected_point_count = len(result.points)
    result.inlier_count = len(result.points)

    line_pts = np.asarray(result.points, dtype=np.float64) if result.points else fit_pts
    y_min = float(np.min(line_pts[:, 1]))
    y_max = float(np.max(line_pts[:, 1]))
    x_min = float(m * y_min + b)
    x_max = float(m * y_max + b)
    result.angle_horizontal, result.angle_vertical = _line_angles(x_min, y_min, x_max, y_max)
    result.fit_line = (x_min, y_min, x_max, y_max)
    return result


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    scan, selected = LimitPeakBoundaryEngine(settings).scan(gray, roi, side)
    result = _result_with_scan_metadata(side, scan)
    return _fit_selected_boundary(result, selected, roi, settings)


def measure_single_taper(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> MeasurementResult:
    taper = measure_taper_side(gray, roi, side, settings)
    result = MeasurementResult(
        measurement_type="taper_single",
        overall_confidence=taper.confidence,
        status=taper.status,
        warning_message=taper.warning_message,
    )
    if side == "right":
        result.right_taper = taper
    else:
        result.left_taper = taper
    return result


def measure_double_taper(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    left = measure_taper_side(gray, roi, "left", settings)
    right = measure_taper_side(gray, roi, "right", settings)
    valid = [taper for taper in (left, right) if taper.status != "Fail"]
    overall = float(np.mean([taper.confidence for taper in valid])) if valid else 0.0
    angles = [taper.angle_horizontal for taper in valid if taper.angle_horizontal is not None]
    avg = float(np.mean(angles)) if angles else None
    diff = (
        abs(left.angle_horizontal - right.angle_horizontal)
        if left.angle_horizontal is not None and right.angle_horizontal is not None
        else None
    )
    status = "Fail" if not valid else "OK" if overall >= MIN_COVERAGE_FOR_OK * 100.0 else "Check"
    return MeasurementResult(
        measurement_type="taper_double",
        left_taper=left,
        right_taper=right,
        avg_taper_angle=avg,
        taper_angle_diff=diff,
        overall_confidence=overall,
        status=status,
        warning_message="; ".join([msg for msg in [left.warning_message, right.warning_message] if msg]),
    )
