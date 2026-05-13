from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from fib_sem_measurement_tool.models.result import HoleCDResult, MeasurementResult, MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


@dataclass(frozen=True)
class HoleEdgeCandidate:
    theta_index: int
    theta: float
    radius: float
    strength: float
    band_start: int
    band_end: int
    band_center: float
    band_peak: float


@dataclass
class HoleBoundaryTrack:
    radii: np.ndarray
    strengths: np.ndarray
    coverage: float = 0.0
    mean_radius: float = 0.0
    radius_std: float = 0.0
    mean_strength: float = 0.0
    smoothness: float = 0.0
    continuity: float = 0.0
    gap_count: int = 0
    max_gap: int = 0
    score: float = 0.0
    warnings: list[str] = field(default_factory=list)


ANGLE_SAMPLES = 360
FERET_SAMPLES = 180


def _refine_center(gray: np.ndarray, roi: Sequence[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [int(v) for v in roi]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    crop = gray[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return cx, cy
    threshold = float(np.percentile(crop, 20.0))
    mask = crop <= threshold
    if int(mask.sum()) < max(20, int(crop.size * 0.015)):
        return cx, cy
    ys, xs = np.nonzero(mask)
    refined_x = x1 + float(np.mean(xs))
    refined_y = y1 + float(np.mean(ys))
    max_shift = min(x2 - x1 + 1, y2 - y1 + 1) * 0.18
    if math.hypot(refined_x - cx, refined_y - cy) > max_shift:
        return cx, cy
    return refined_x, refined_y


def _max_radius_to_roi(cx: float, cy: float, roi: Sequence[int]) -> int:
    x1, y1, x2, y2 = [int(v) for v in roi]
    return max(1, int(math.floor(min(cx - x1, x2 - cx, cy - y1, y2 - cy))))


def _polar_image(gray: np.ndarray, center: tuple[float, float], max_radius: int, angle_count: int = ANGLE_SAMPLES) -> np.ndarray:
    radii = np.arange(max_radius + 1, dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False, dtype=np.float32)
    xs = center[0] + np.cos(theta)[:, None] * radii[None, :]
    ys = center[1] + np.sin(theta)[:, None] * radii[None, :]
    return cv2.remap(
        gray.astype(np.float32),
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _candidate_bands(profile: np.ndarray, gradient: np.ndarray, limit: float, min_radius: int, max_radius: int) -> list[tuple[int, int, int, float]]:
    candidates: list[tuple[int, int, int, float]] = []
    for idx in range(max(1, min_radius), min(max_radius, gradient.size - 1)):
        strength = float(gradient[idx])
        if strength < limit or strength < gradient[idx - 1] or strength < gradient[idx + 1]:
            continue
        half = max(limit * 0.45, strength * 0.45)
        start = idx
        while start > min_radius and gradient[start - 1] >= half:
            start -= 1
        end = idx
        while end < max_radius and gradient[end + 1] >= half:
            end += 1
        width = end - start + 1
        if width < 1 or width > max(18, int(profile.size * 0.12)):
            continue
        candidates.append((start, end, idx, strength))
    return candidates


def _extract_candidates(polar: np.ndarray, settings: MeasurementSettings) -> list[list[HoleEdgeCandidate]]:
    blurred = cv2.GaussianBlur(polar, (1, 5), 0)
    gradient = np.abs(np.gradient(blurred, axis=1))
    max_radius = polar.shape[1] - 1
    min_radius = max(6, int(max_radius * 0.08))
    outer_radius = max(min_radius + 3, int(max_radius * 0.92))
    finite = gradient[:, min_radius:outer_radius]
    dynamic_limit = float(np.percentile(finite, 78.0)) if finite.size else 0.0
    user_limit = float(getattr(settings, "minimum_grayscale_delta", 55.0)) * 0.12
    limit = max(3.0, min(max(dynamic_limit, user_limit), float(np.percentile(finite, 92.0)) if finite.size else 255.0))

    by_angle: list[list[HoleEdgeCandidate]] = []
    for theta_index in range(polar.shape[0]):
        row_candidates = []
        for start, end, peak, strength in _candidate_bands(polar[theta_index], gradient[theta_index], limit, min_radius, outer_radius):
            theta = 2.0 * math.pi * theta_index / polar.shape[0]
            row_candidates.append(
                HoleEdgeCandidate(
                    theta_index=theta_index,
                    theta=theta,
                    radius=float(peak),
                    strength=float(strength),
                    band_start=start,
                    band_end=end,
                    band_center=float((start + end) * 0.5),
                    band_peak=float(peak),
                )
            )
        by_angle.append(row_candidates)
    return by_angle


def _circular_gaps(valid: np.ndarray) -> tuple[int, int]:
    if bool(np.all(valid)):
        return 0, 0
    doubled = np.concatenate([valid, valid])
    gaps: list[int] = []
    run = 0
    for item in doubled:
        if not item:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if run:
        gaps.append(run)
    max_gap = min(max(gaps) if gaps else 0, valid.size)
    starts = int(np.sum((~valid) & np.roll(valid, 1)))
    return starts, max_gap


def _interp_circular(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return values
    n = values.size
    xp = np.flatnonzero(finite)
    fp = values[finite]
    xp_ext = np.concatenate([xp - n, xp, xp + n])
    fp_ext = np.concatenate([fp, fp, fp])
    return np.interp(np.arange(n), xp_ext, fp_ext)


def _smooth_circular(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _score_track(radii: np.ndarray, strengths: np.ndarray, max_radius: int) -> HoleBoundaryTrack:
    valid = np.isfinite(radii)
    coverage = float(valid.mean()) if valid.size else 0.0
    gap_count, max_gap = _circular_gaps(valid)
    filled = _interp_circular(radii)
    finite = filled[np.isfinite(filled)]
    if finite.size == 0:
        return HoleBoundaryTrack(radii, strengths, warnings=["no_valid_hole_boundary"])
    smoothed = _smooth_circular(filled, 5)
    deltas = np.abs(np.diff(np.concatenate([smoothed, smoothed[:1]])))
    mean_radius = float(np.mean(finite))
    radius_std = float(np.std(finite))
    mean_strength = float(np.nanmean(strengths)) if np.isfinite(strengths).any() else 0.0
    smoothness = float(np.mean(deltas))
    continuity = max(0.0, 1.0 - float(max_gap) / max(1.0, radii.size * 0.18))
    radius_cv = radius_std / max(mean_radius, 1.0)
    warnings = []
    if coverage < 0.45:
        warnings.append("coverage_too_low")
    if mean_radius < max_radius * 0.12:
        warnings.append("boundary_too_close_to_center")
    if mean_radius > max_radius * 0.88:
        warnings.append("boundary_too_close_to_roi_edge")
    if max_gap > radii.size * 0.16:
        warnings.append("too_many_gaps")
    if mean_strength < 3.0:
        warnings.append("low_edge_strength")
    if radius_cv > 0.32:
        warnings.append("possible_inner_stain_detected")
    score = (
        coverage * 45.0
        + min(mean_strength / 35.0, 1.0) * 20.0
        + continuity * 20.0
        + max(0.0, 1.0 - min(radius_cv / 0.25, 1.0)) * 10.0
        + max(0.0, 1.0 - min(smoothness / max(2.0, mean_radius * 0.05), 1.0)) * 5.0
    )
    return HoleBoundaryTrack(
        radii=smoothed,
        strengths=strengths,
        coverage=coverage,
        mean_radius=mean_radius,
        radius_std=radius_std,
        mean_strength=mean_strength,
        smoothness=smoothness,
        continuity=continuity,
        gap_count=gap_count,
        max_gap=max_gap,
        score=float(score),
        warnings=warnings,
    )


def _build_tracks(candidates_by_angle: list[list[HoleEdgeCandidate]], max_radius: int) -> list[HoleBoundaryTrack]:
    all_radii = [candidate.radius for row in candidates_by_angle for candidate in row]
    if not all_radii:
        return []
    hist, edges = np.histogram(all_radii, bins=max(24, min(80, max_radius // 3)), range=(0, max_radius))
    peak_bins = np.argsort(hist)[::-1][:10]
    seeds = [float((edges[idx] + edges[idx + 1]) * 0.5) for idx in peak_bins if hist[idx] >= max(8, len(candidates_by_angle) * 0.05)]
    tracks = []
    for seed in seeds:
        radii = np.full(len(candidates_by_angle), np.nan, dtype=np.float64)
        strengths = np.full(len(candidates_by_angle), np.nan, dtype=np.float64)
        previous = seed
        max_jump = max(8.0, seed * 0.16)
        for idx, row in enumerate(candidates_by_angle):
            if not row:
                continue
            candidate = min(row, key=lambda item: abs(item.radius - previous))
            if abs(candidate.radius - previous) <= max_jump:
                radii[idx] = candidate.radius
                strengths[idx] = candidate.strength
                previous = candidate.radius
        # second pass helps close wrap-around gaps after a seed has stabilized.
        filled = _interp_circular(radii)
        for idx, row in enumerate(candidates_by_angle):
            if np.isfinite(radii[idx]) or not row or not np.isfinite(filled[idx]):
                continue
            candidate = min(row, key=lambda item: abs(item.radius - filled[idx]))
            if abs(candidate.radius - filled[idx]) <= max_jump:
                radii[idx] = candidate.radius
                strengths[idx] = candidate.strength
        track = _score_track(radii, strengths, max_radius)
        if track.coverage <= 0:
            continue
        if any(abs(track.mean_radius - other.mean_radius) < max(5.0, max_radius * 0.035) for other in tracks):
            continue
        tracks.append(track)
    return tracks


def _contour_from_track(track: HoleBoundaryTrack, center: tuple[float, float]) -> np.ndarray:
    radii = _interp_circular(track.radii)
    if not np.isfinite(radii).all():
        return np.empty((0, 2), dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, radii.size, endpoint=False)
    x = center[0] + radii * np.cos(theta)
    y = center[1] + radii * np.sin(theta)
    return np.column_stack([x, y]).astype(np.float64)


def _polygon_area(contour: np.ndarray) -> float:
    x = contour[:, 0]
    y = contour[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _perimeter(contour: np.ndarray) -> float:
    diff = np.diff(np.vstack([contour, contour[:1]]), axis=0)
    return float(np.sum(np.hypot(diff[:, 0], diff[:, 1])))


def _feret(contour: np.ndarray) -> tuple[float, float]:
    widths = []
    for angle in np.linspace(0.0, np.pi, FERET_SAMPLES, endpoint=False):
        projection = contour[:, 0] * math.cos(angle) + contour[:, 1] * math.sin(angle)
        widths.append(float(np.max(projection) - np.min(projection)))
    return float(np.min(widths)), float(np.max(widths))


def _ellipse_metrics(contour: np.ndarray) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if contour.shape[0] < 5:
        return None, None, None, None, None
    points = contour.astype(np.float32).reshape(-1, 1, 2)
    try:
        (_center, axes, angle) = cv2.fitEllipse(points)
    except cv2.error:
        return None, None, None, None, None
    major = float(max(axes))
    minor = float(min(axes))
    ellipse_points = []
    cx, cy = _center
    a = major * 0.5
    b = minor * 0.5
    phi = math.radians(float(angle))
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    for x, y in contour:
        dx, dy = x - cx, y - cy
        ex = dx * cos_phi + dy * sin_phi
        ey = -dx * sin_phi + dy * cos_phi
        ellipse_points.append(abs(math.hypot(ex / max(a, 1e-6), ey / max(b, 1e-6)) - 1.0))
    return major, minor, float(angle), float(np.mean(ellipse_points)), major / max(minor, 1e-6)


def _result_from_track(track: HoleBoundaryTrack, contour: np.ndarray, center: tuple[float, float], target: str) -> HoleCDResult:
    if contour.shape[0] < 16:
        return HoleCDResult(target=target, status=MeasurementStatus.FAIL.value, warning_message="contour_reconstruction_failed")
    area = _polygon_area(contour)
    perimeter = _perimeter(contour)
    horizontal = float(np.max(contour[:, 0]) - np.min(contour[:, 0]))
    vertical = float(np.max(contour[:, 1]) - np.min(contour[:, 1]))
    min_feret, max_feret = _feret(contour)
    equivalent = float(math.sqrt(4.0 * area / math.pi)) if area > 0 else None
    major, minor, angle, fit_error, aspect = _ellipse_metrics(contour)
    warnings = list(track.warnings)
    if area <= 4.0 or perimeter <= 4.0:
        warnings.append("contour_reconstruction_failed")
    confidence = max(0.0, min(100.0, track.score - len(warnings) * 8.0 - (fit_error or 0.0) * 8.0))
    if warnings and confidence < 45.0:
        status = MeasurementStatus.REVIEW_NEEDED.value
    elif confidence >= 80.0 and not warnings:
        status = MeasurementStatus.OK.value
    elif confidence >= 58.0:
        status = MeasurementStatus.CHECK.value
    else:
        status = MeasurementStatus.REVIEW_NEEDED.value
    if "contour_reconstruction_failed" in warnings:
        status = MeasurementStatus.FAIL.value
    return HoleCDResult(
        target=target,
        horizontal_px=horizontal,
        vertical_px=vertical,
        min_feret_px=min_feret,
        max_feret_px=max_feret,
        equivalent_diameter_px=equivalent,
        area_px=area,
        perimeter_px=perimeter,
        coverage=track.coverage,
        mean_radius=track.mean_radius,
        radius_std=track.radius_std,
        mean_strength=track.mean_strength,
        smoothness=track.smoothness,
        continuity=track.continuity,
        gap_count=track.gap_count,
        max_gap=track.max_gap,
        confidence=confidence,
        status=status,
        warning_message=";".join(dict.fromkeys(warnings)),
        contour_points=[(float(x), float(y)) for x, y in contour],
        center=(float(center[0]), float(center[1])),
        ellipse_major_px=major,
        ellipse_minor_px=minor,
        ellipse_angle_deg=angle,
        ellipse_fit_error=fit_error,
        ellipse_aspect_ratio=aspect,
    )


def measure_hole_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    center = _refine_center(gray, roi)
    max_radius = _max_radius_to_roi(center[0], center[1], roi)
    target = getattr(settings, "hole_target", "inner")
    if max_radius < 12:
        hole = HoleCDResult(target=target, status=MeasurementStatus.FAIL.value, warning_message="roi_too_small")
        return MeasurementResult(measurement_type="hole_cd", hole_cd=hole, overall_confidence=0.0, status=MeasurementStatus.FAIL.value)
    polar = _polar_image(gray, center, max_radius, ANGLE_SAMPLES)
    candidates = _extract_candidates(polar, settings)
    tracks = _build_tracks(candidates, max_radius)
    valid_tracks = [
        track
        for track in tracks
        if track.coverage >= 0.45
        and track.mean_radius >= max_radius * 0.12
        and track.mean_radius <= max_radius * 0.88
        and track.max_gap <= ANGLE_SAMPLES * 0.20
        and track.score >= 38.0
    ]
    if not valid_tracks:
        hole = HoleCDResult(target=target, status=MeasurementStatus.FAIL.value, warning_message="no_valid_hole_boundary")
        return MeasurementResult(measurement_type="hole_cd", hole_cd=hole, overall_confidence=0.0, status=MeasurementStatus.FAIL.value)
    valid_tracks.sort(key=lambda track: track.mean_radius)
    selected = valid_tracks[-1] if target == "outer" else valid_tracks[0]
    if len(valid_tracks) > 1 and abs(valid_tracks[-1].mean_radius - valid_tracks[0].mean_radius) < max(6.0, max_radius * 0.08):
        selected.warnings.append("ambiguous_concentric_boundaries")
    contour = _contour_from_track(selected, center)
    hole = _result_from_track(selected, contour, center, target)
    return MeasurementResult(
        measurement_type="hole_cd",
        hole_cd=hole,
        overall_confidence=hole.confidence,
        status=hole.status,
        warning_message=hole.warning_message,
    )
