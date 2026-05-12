from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    detect_profile_candidates,
    prepare_display_profile_signal,
)
from fib_sem_measurement_tool.models.result import EllipseCDResult, MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


RAY_COUNT = 16
MIN_POINT_COUNT = 5
MIN_CENTER_RADIUS_RATIO = 0.08
MAX_EDGE_RADIUS_MARGIN_PX = 1.5
PERSISTENCE_WINDOW = 5
TAIL_PERSISTENCE_WINDOW = 12


def _ray_limit(cx: float, cy: float, dx: float, dy: float, roi: Sequence[int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in roi]
    limits = []
    if dx > 1e-6:
        limits.append((x2 - cx) / dx)
    elif dx < -1e-6:
        limits.append((x1 - cx) / dx)
    if dy > 1e-6:
        limits.append((y2 - cy) / dy)
    elif dy < -1e-6:
        limits.append((y1 - cy) / dy)
    positive = [value for value in limits if value > 0.0]
    return float(min(positive)) if positive else 0.0


def _sample_profile(gray: np.ndarray, cx: float, cy: float, dx: float, dy: float, max_radius: float) -> tuple[np.ndarray, np.ndarray]:
    radii = np.arange(0.0, max_radius + 0.5, 1.0, dtype=np.float32)
    if radii.size < 3:
        return radii, np.asarray([], dtype=np.float32)
    xs = np.clip(cx + radii * dx, 0.0, float(gray.shape[1] - 1)).astype(np.float32)
    ys = np.clip(cy + radii * dy, 0.0, float(gray.shape[0] - 1)).astype(np.float32)
    profile = cv2.remap(
        gray.astype(np.float32, copy=False),
        xs.reshape(1, -1),
        ys.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return radii, profile.reshape(-1).astype(np.float32)


def _candidate_settings(settings: MeasurementSettings, min_delta: float) -> MeasurementSettings:
    candidate_settings = settings.clone()
    candidate_settings.minimum_grayscale_delta = float(min_delta)
    candidate_settings.denoise_grayscale_profiles = True
    candidate_settings.profile_denoise_window = max(7, int(getattr(settings, "profile_denoise_window", 7)))
    candidate_settings.profile_denoise_range_sigma = max(
        18.0,
        float(getattr(settings, "profile_denoise_range_sigma", 28.0)),
    )
    return candidate_settings


def _profile_transition_hint(signal: np.ndarray, min_delta: float) -> float | None:
    window = min(max(5, signal.size // 8), max(5, signal.size // 4))
    if signal.size < window * 2:
        return None
    inner_mean = float(np.mean(signal[:window], dtype=np.float64))
    outer_mean = float(np.mean(signal[-window:], dtype=np.float64))
    contrast = outer_mean - inner_mean
    if abs(contrast) < max(8.0, float(min_delta) * 0.5):
        return None
    return contrast


def _persistent_transition(
    signal: np.ndarray,
    position: float,
    signed_delta: float,
    min_delta: float,
    expected_contrast: float | None,
) -> bool:
    center = int(round(float(position)))
    candidate_sign = np.sign(float(signed_delta))
    if (
        expected_contrast is not None
        and candidate_sign != 0.0
        and candidate_sign != np.sign(expected_contrast)
    ):
        return False
    left_start = max(0, center - PERSISTENCE_WINDOW)
    left_end = max(left_start, center)
    right_start = min(signal.size, center + 1)
    right_end = min(signal.size, right_start + PERSISTENCE_WINDOW)
    if left_end <= left_start or right_end <= right_start:
        return False
    before = float(np.mean(signal[left_start:left_end], dtype=np.float64))
    after = float(np.mean(signal[right_start:right_end], dtype=np.float64))
    contrast = after - before
    if abs(contrast) < max(5.0, float(min_delta) * 0.35):
        return False
    if contrast != 0.0 and np.sign(contrast) != np.sign(float(signed_delta)):
        return False

    tail_left_end = max(0, center - 2)
    tail_left_start = max(0, tail_left_end - TAIL_PERSISTENCE_WINDOW)
    tail_right_start = min(signal.size, center + 2)
    tail_right_end = min(signal.size, tail_right_start + TAIL_PERSISTENCE_WINDOW)
    if tail_left_end - tail_left_start >= 4 and tail_right_end - tail_right_start >= 4:
        tail_before = float(np.mean(signal[tail_left_start:tail_left_end], dtype=np.float64))
        tail_after = float(np.mean(signal[tail_right_start:tail_right_end], dtype=np.float64))
        tail_contrast = tail_after - tail_before
        tail_min_contrast = max(7.0, float(min_delta) * 0.45)
        if expected_contrast is not None:
            tail_min_contrast = max(tail_min_contrast, abs(expected_contrast) * 0.35)
        if abs(tail_contrast) < tail_min_contrast:
            return False
        if tail_contrast != 0.0 and np.sign(tail_contrast) != np.sign(float(signed_delta)):
            return False
    return True


def _first_edge_on_ray(
    profile: np.ndarray,
    radii: np.ndarray,
    max_radius: float,
    settings: MeasurementSettings,
) -> tuple[float, float] | None:
    if profile.size < 3:
        return None
    min_radius = max(3.0, max_radius * MIN_CENTER_RADIUS_RATIO)
    max_valid_radius = max(0.0, max_radius - MAX_EDGE_RADIUS_MARGIN_PX)
    threshold = max(1.0, float(getattr(settings, "minimum_grayscale_delta", 30.0)))
    thresholds = [threshold, max(8.0, threshold * 0.55)]
    positions = np.arange(profile.size, dtype=np.float32)

    for candidate_threshold in thresholds:
        candidate_settings = _candidate_settings(settings, candidate_threshold)
        signal = prepare_display_profile_signal(profile, "horizontal", candidate_settings)
        expected_contrast = _profile_transition_hint(signal, candidate_threshold)
        candidates = detect_profile_candidates(
            profile,
            candidate_settings,
            scan_axis="horizontal",
            scan_index=0,
            local_scan_index=0,
            roi_origin=(0, 0),
            detection_profile=signal,
        )
        for candidate in sorted(candidates, key=lambda item: float(item.position)):
            radius = float(np.interp(float(candidate.position), positions, radii))
            if radius < min_radius:
                continue
            if radius > max_valid_radius:
                continue
            if not _persistent_transition(
                signal,
                candidate.position,
                candidate.signed_delta,
                candidate_threshold,
                expected_contrast,
            ):
                continue
            return radius, float(candidate.strength)
    return None


def _remove_radius_outliers(points: list[tuple[float, float]], center: tuple[float, float], settings: MeasurementSettings) -> tuple[list[tuple[float, float]], int, float]:
    if len(points) < MIN_POINT_COUNT:
        return points, 0, 0.0
    cx, cy = center
    radii = np.asarray([math.hypot(x - cx, y - cy) for x, y in points], dtype=np.float64)
    median = float(np.median(radii))
    mad = float(np.median(np.abs(radii - median)))
    robust_sigma = 1.4826 * mad
    max_jump = float(getattr(settings, "max_jump_px", 28.0))
    limit = max(max_jump, robust_sigma * 3.5, 3.0)
    kept = [point for point, radius in zip(points, radii) if abs(float(radius) - median) <= limit]
    return kept, len(points) - len(kept), float(np.std(radii) / max(1.0, median))


def _ellipse_diameters(ellipse) -> tuple[float, float]:
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    a = float(axis_a) * 0.5
    b = float(axis_b) * 0.5
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xs = float(cx) + a * np.cos(theta) * cos_a - b * np.sin(theta) * sin_a
    ys = float(cy) + a * np.cos(theta) * sin_a + b * np.sin(theta) * cos_a
    return float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys))


def _score_result(result: EllipseCDResult, strength_values: list[float]) -> None:
    if result.valid_point_count < MIN_POINT_COUNT:
        result.confidence = 0.0
        result.status = "Fail"
        return
    success_ratio = result.valid_point_count / max(1, result.ray_attempt_count)
    outlier_ratio = result.outlier_count / max(1, result.valid_point_count + result.outlier_count)
    radius_stability = 1.0 - min(1.0, float(result.radius_cv or 0.0) * 2.0)
    strength_score = min(1.0, float(np.mean(strength_values)) / 80.0) if strength_values else 0.5
    fit_score = min(1.0, result.valid_point_count / 10.0)
    confidence = 100.0 * (
        0.35 * success_ratio
        + 0.20 * (1.0 - outlier_ratio)
        + 0.20 * radius_stability
        + 0.15 * strength_score
        + 0.10 * fit_score
    )
    result.confidence = float(max(0.0, min(100.0, confidence)))
    if result.valid_point_count >= 10 and result.confidence >= 80.0:
        result.status = "OK"
    elif result.valid_point_count >= 8 and result.confidence >= 60.0:
        result.status = "Check"
    elif result.valid_point_count >= MIN_POINT_COUNT:
        result.status = "Review Needed"
    else:
        result.status = "Fail"


def measure_ellipse_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    x1, y1, x2, y2 = [int(v) for v in roi]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    points: list[tuple[float, float]] = []
    strengths: list[float] = []

    for index in range(RAY_COUNT):
        angle = 2.0 * math.pi * index / RAY_COUNT
        dx = math.cos(angle)
        dy = math.sin(angle)
        max_radius = _ray_limit(cx, cy, dx, dy, roi)
        radii, profile = _sample_profile(gray, cx, cy, dx, dy, max_radius)
        edge = _first_edge_on_ray(profile, radii, max_radius, settings)
        if edge is None:
            continue
        radius, strength = edge
        points.append((float(cx + radius * dx), float(cy + radius * dy)))
        strengths.append(strength)

    kept_points, outlier_count, radius_cv = _remove_radius_outliers(points, (cx, cy), settings)
    result = EllipseCDResult(
        ray_attempt_count=RAY_COUNT,
        valid_point_count=len(kept_points),
        outlier_count=outlier_count,
        boundary_points=kept_points,
        edge_strength_mean=float(np.mean(strengths)) if strengths else None,
        radius_cv=radius_cv,
    )
    if len(kept_points) < MIN_POINT_COUNT:
        result.status = "Fail"
        result.warning_message = "ellipse cd valid points fewer than 5"
        return MeasurementResult(measurement_type="ellipse_cd", ellipse_cd=result, warning_message=result.warning_message)

    try:
        fit_input = np.asarray(kept_points, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(fit_input)
    except cv2.error as exc:
        result.status = "Fail"
        result.warning_message = f"ellipse cd fit failed: {exc}"
        return MeasurementResult(measurement_type="ellipse_cd", ellipse_cd=result, warning_message=result.warning_message)

    (fit_cx, fit_cy), (axis_a, axis_b), angle_deg = ellipse
    result.center_x = float(fit_cx)
    result.center_y = float(fit_cy)
    result.major_axis_px = float(max(axis_a, axis_b))
    result.minor_axis_px = float(min(axis_a, axis_b))
    result.angle_deg = float(angle_deg)
    result.horizontal_diameter_px, result.vertical_diameter_px = _ellipse_diameters(ellipse)
    _score_result(result, strengths)
    if result.status != "OK":
        result.warning_message = "ellipse cd needs review"
    return MeasurementResult(
        measurement_type="ellipse_cd",
        ellipse_cd=result,
        overall_confidence=result.confidence,
        status=result.status,
        warning_message=result.warning_message,
    )
