from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.models.settings import MeasurementSettings


@dataclass
class EdgeBand:
    start: float
    end: float
    center: float
    peak: float
    strength: float
    sign: int
    width: float
    scan_index: int
    scan_coord: float
    axis_coord: float


@dataclass
class BoundaryTrack:
    points: List[Tuple[float, float]]
    side: str
    coverage: float
    mean_strength: float
    smoothness: float
    continuity: float
    fit_error: float
    score: float


def _smooth_profile(profile: np.ndarray, settings: MeasurementSettings) -> np.ndarray:
    data = profile.astype(np.float32, copy=False).reshape(-1)
    if data.size < 5:
        return data
    k = max(1, int(settings.advanced.blur_kernel))
    if k % 2 == 0:
        k += 1
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        data = np.convolve(data, kernel, mode="same")
    m = max(0, int(settings.advanced.median_filter_size))
    if m > 1 and m % 2 == 1:
        pad = m // 2
        padded = np.pad(data, (pad, pad), mode="edge")
        data = np.array([np.median(padded[i : i + m]) for i in range(data.size)], dtype=np.float32)
    return data


def extract_edge_bands(
    profile: Sequence[float],
    scan_index: int,
    scan_coord: float,
    axis_offset: float,
    settings: MeasurementSettings,
) -> List[EdgeBand]:
    data = _smooth_profile(np.asarray(profile, dtype=np.float32), settings)
    if data.size < 8:
        return []
    grad = np.gradient(data)
    mag = np.abs(grad)
    std = float(np.std(mag))
    med = float(np.median(mag))
    mx = float(np.max(mag))
    if mx <= 1e-6:
        return []
    thr = med + float(settings.advanced.sensitivity) * std
    thr = max(thr, float(settings.advanced.peak_prominence) * mx)
    min_strength = max(1e-6, 0.35 * thr)
    margin = max(2, int(round(data.size * 0.02)))

    raw = []
    for i in range(1, data.size - 1):
        if i < margin or i > data.size - margin - 1:
            continue
        if mag[i] < thr:
            continue
        if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
            raw.append((float(i), float(mag[i]), 1 if grad[i] >= 0 else -1))
    raw = [item for item in raw if item[1] >= min_strength]
    if not raw:
        return []

    grouped: List[List[Tuple[float, float, int]]] = []
    gap = max(2.0, data.size * 0.02)
    for item in raw:
        if not grouped or (item[0] - grouped[-1][-1][0] > gap) or (grouped[-1][-1][2] != item[2]):
            grouped.append([item])
        else:
            grouped[-1].append(item)

    bands: List[EdgeBand] = []
    for g in grouped:
        peak = max(g, key=lambda x: x[1])
        start, end = g[0][0], g[-1][0]
        center = (start + end) / 2.0
        peak_pos = peak[0]
        bands.append(
            EdgeBand(
                start=start,
                end=end,
                center=center,
                peak=peak_pos,
                strength=float(peak[1]),
                sign=int(peak[2]),
                width=float(end - start),
                scan_index=int(scan_index),
                scan_coord=float(scan_coord),
                axis_coord=float(axis_offset + peak_pos),
            )
        )
    return bands


def edge_position(band: EdgeBand, edge_reference: str, side: str) -> float:
    if edge_reference == "center":
        return band.center
    if edge_reference == "strongest":
        return band.peak
    if edge_reference == "outer":
        return band.start if side in {"left", "top"} else band.end
    if edge_reference == "inner":
        return band.end if side in {"left", "top"} else band.start
    return band.peak


def build_boundary_track(
    scan_bands: List[List[EdgeBand]], side: str, edge_reference: str, settings: MeasurementSettings
) -> Optional[BoundaryTrack]:
    total = len(scan_bands)
    if total == 0:
        return None
    center_prior = 0.25 if side in {"left", "top"} else 0.75
    max_jump = 8.0
    tracks: List[Dict[str, object]] = []
    for idx, bands in enumerate(scan_bands):
        for band in bands:
            pos = edge_position(band, edge_reference, side)
            x = pos
            tracks.append({"points": [(idx, x, band)], "last": x, "gaps": 0})
        new_tracks = []
        for tr in tracks:
            last = float(tr["last"])
            attached = False
            for band in bands:
                pos = edge_position(band, edge_reference, side)
                if abs(pos - last) <= max_jump:
                    pts = list(tr["points"])
                    pts.append((idx, pos, band))
                    new_tracks.append({"points": pts, "last": pos, "gaps": int(tr["gaps"])})
                    attached = True
            if not attached and int(tr["gaps"]) < 2:
                new_tracks.append({"points": tr["points"], "last": tr["last"], "gaps": int(tr["gaps"]) + 1})
        if new_tracks:
            tracks = sorted(new_tracks, key=lambda t: len(t["points"]), reverse=True)[:50]

    min_points = max(3, int(settings.advanced.minimum_valid_line_count * 0.6), int(total * settings.advanced.min_valid_line_ratio))
    best: Optional[BoundaryTrack] = None
    for tr in tracks:
        pts = tr["points"]
        if len(pts) < min_points:
            continue
        scan_coords = np.array([p[2].scan_coord for p in pts], dtype=np.float32)
        edge_coords = np.array([p[2].axis_coord - p[2].peak + p[1] for p in pts], dtype=np.float32)
        strengths = np.array([p[2].strength for p in pts], dtype=np.float32)
        coverage = float(len(pts) / max(1, total))
        diffs = np.diff([p[1] for p in pts])
        smoothness = float(1.0 / (1.0 + np.std(diffs))) if diffs.size else 1.0
        continuity = float(1.0 - min(1.0, int(tr["gaps"]) / max(1, total)))
        slope, inter = np.polyfit(scan_coords, edge_coords, 1)
        fit = slope * scan_coords + inter
        fit_error = float(np.sqrt(np.mean((edge_coords - fit) ** 2)))
        mean_strength = float(np.mean(strengths))
        prior_pos = float(np.median([p[1] for p in pts]))
        prior_score = max(0.0, 1.0 - abs(prior_pos / max(1.0, np.max([len(b) for b in scan_bands if b] + [1])) - center_prior))
        score = 100.0 * (0.32 * coverage + 0.18 * min(1.0, mean_strength / (np.max(strengths) + 1e-6)) + 0.15 * smoothness + 0.15 * continuity + 0.10 * prior_score + 0.10 * (1.0 / (1.0 + fit_error)))
        candidate = BoundaryTrack(
            points=[(float(sc), float(ec)) for sc, ec in zip(scan_coords, edge_coords)],
            side=side,
            coverage=coverage,
            mean_strength=mean_strength,
            smoothness=smoothness,
            continuity=continuity,
            fit_error=fit_error,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def interpolate_track(track: BoundaryTrack, at: np.ndarray) -> np.ndarray:
    coords = np.asarray(track.points, dtype=np.float32)
    if coords.shape[0] < 2:
        return np.full_like(at, np.nan, dtype=np.float32)
    order = np.argsort(coords[:, 0])
    return np.interp(at, coords[order, 0], coords[order, 1], left=np.nan, right=np.nan)
