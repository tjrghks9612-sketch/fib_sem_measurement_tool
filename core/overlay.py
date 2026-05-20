from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fib_sem_measurement_tool.models.result import (
    DistanceResult,
    EllipseCDResult,
    MeasurementResult,
    TaperSideResult,
)
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import measurement_label, status_label, t, taper_side_label


Color = Tuple[int, int, int]

BG_PANEL: Color = (15, 20, 27)
TEXT: Color = (244, 248, 252)
MUTED: Color = (164, 176, 188)
ROI_COLOR: Color = (112, 193, 255)
CD_COLOR: Color = (64, 196, 255)
THK_COLOR: Color = (112, 234, 178)
TAPER_LEFT_COLOR: Color = (255, 216, 111)
TAPER_RIGHT_COLOR: Color = (255, 155, 111)
TAPER_HEIGHT_COLOR: Color = (190, 178, 255)
ELLIPSE_COLOR: Color = (92, 211, 255)
POINT_COLOR: Color = (244, 248, 252)
FAIL_COLOR: Color = (95, 125, 255)
SHADOW: Color = (8, 10, 14)


@lru_cache(maxsize=12)
def _ui_font(size: int):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_text(text: str, font) -> Tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _draw_text(image: np.ndarray, text: str, origin: Tuple[int, int], color: Color, font_size: int = 16) -> None:
    font = _ui_font(font_size)
    x, y = origin
    tw, th = _measure_text(text, font)
    x = max(0, min(int(x), image.shape[1] - 1))
    y = max(0, min(int(y), image.shape[0] - 1))
    x2 = min(image.shape[1], x + tw + 4)
    y2 = min(image.shape[0], y + th + 6)
    if x2 <= x or y2 <= y:
        return
    patch = image[y:y2, x:x2]
    pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    draw.text((0, 0), text, fill=rgb, font=font)
    image[y:y2, x:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_text_centered(image: np.ndarray, text: str, box: Tuple[int, int, int, int], color: Color, font_size: int = 16) -> None:
    font = _ui_font(font_size)
    left, top, right, bottom = [int(v) for v in box]
    left = max(0, min(left, image.shape[1] - 1))
    top = max(0, min(top, image.shape[0] - 1))
    right = max(left + 1, min(right, image.shape[1]))
    bottom = max(top + 1, min(bottom, image.shape[0]))
    if right <= left or bottom <= top:
        return
    patch = image[top:bottom, left:right]
    if patch.size == 0:
        return
    pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    lines = text.splitlines() or [text]
    line_sizes = [_measure_text(line, font) for line in lines]
    line_gap = max(3, font_size // 5)
    total_h = sum(size[1] for size in line_sizes) + line_gap * (len(lines) - 1)
    y = max(0, int(round((patch.shape[0] - total_h) * 0.5)))
    for line, (tw, th) in zip(lines, line_sizes):
        x = max(0, int(round((patch.shape[1] - tw) * 0.5)))
        draw.text((x, y), line, fill=rgb, font=font)
        y += th + line_gap
    image[top:bottom, left:right] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _panel(image: np.ndarray, left: int, top: int, right: int, bottom: int, alpha: float = 0.72) -> None:
    left = max(0, min(int(left), image.shape[1] - 1))
    top = max(0, min(int(top), image.shape[0] - 1))
    right = max(left + 1, min(int(right), image.shape[1]))
    bottom = max(top + 1, min(int(bottom), image.shape[0]))
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), BG_PANEL, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def _stroke_line(
    image: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Color,
    thickness: int = 2,
    shadow: int = 4,
) -> None:
    cv2.line(image, p1, p2, SHADOW, shadow, cv2.LINE_AA)
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def _stroke_line_alpha(
    image: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Color,
    thickness: int = 1,
    alpha: float = 0.45,
) -> None:
    overlay = image.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def _stroke_polyline(image: np.ndarray, points: np.ndarray, closed: bool, color: Color, thickness: int = 2) -> None:
    cv2.polylines(image, [points], closed, SHADOW, thickness + 3, cv2.LINE_AA)
    cv2.polylines(image, [points], closed, color, thickness, cv2.LINE_AA)


def _stroke_trace_segments(
    image: np.ndarray,
    points: Sequence[tuple[float, float]],
    color: Color,
    thickness: int = 1,
    *,
    max_dx: float = 22.0,
    max_dy: float = 3.5,
    min_points: int = 4,
    longest_only: bool = False,
) -> None:
    if len(points) < 2:
        return
    ordered = sorted(((float(x), float(y)) for x, y in points), key=lambda item: (item[1], item[0]))
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    previous: Optional[tuple[float, float]] = None
    for x, y in ordered:
        if previous is not None and (abs(x - previous[0]) > max_dx or abs(y - previous[1]) > max_dy):
            if len(current) >= min_points:
                segments.append(current)
            current = []
        current.append((int(round(x)), int(round(y))))
        previous = (x, y)
    if len(current) >= min_points:
        segments.append(current)
    if longest_only and segments:
        segments = [max(segments, key=len)]
    for segment in segments:
        _stroke_polyline(image, np.asarray(segment, dtype=np.int32).reshape(-1, 1, 2), False, color, thickness)


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _bbox_for_points(points: Sequence[Tuple[int, int]], pad: int = 8) -> Tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Color,
    scale: float = 0.55,
    avoid: Sequence[Tuple[int, int, int, int]] = (),
) -> Tuple[int, int, int, int]:
    font_size = max(11, int(round(scale * 28)))
    font = _ui_font(font_size)
    lines = text.splitlines() or [text]
    sizes = [_measure_text(line, font) for line in lines]
    tw = max(width for width, _height in sizes)
    th = sum(height for _width, height in sizes) + max(3, font_size // 5) * (len(lines) - 1)
    pad_x = 8
    pad_y = 5
    x0, y0 = int(origin[0]), int(origin[1])
    candidates = [
        (x0, y0 - th),
        (x0 + 16, y0 + 16),
        (x0 - tw - 20, y0 - th),
        (x0 - tw - 20, y0 + 16),
        (x0 + 16, y0 - th - 24),
        (x0 - tw - 20, y0 - th - 24),
    ]
    selected = candidates[0]
    best_penalty = float("inf")
    for candidate_x, candidate_y in candidates:
        left = max(4, min(int(candidate_x - pad_x), image.shape[1] - tw - pad_x * 2 - 4))
        top = max(4, min(int(candidate_y - pad_y), image.shape[0] - th - pad_y * 2 - 4))
        box = (left, top, left + tw + pad_x * 2, top + th + pad_y * 2)
        overlap_penalty = sum(1 for avoid_box in avoid if _boxes_overlap(box, avoid_box))
        distance_penalty = abs(left - (x0 - pad_x)) * 0.001 + abs(top - (y0 - th - pad_y)) * 0.001
        penalty = overlap_penalty * 1000.0 + distance_penalty
        if penalty < best_penalty:
            best_penalty = penalty
            selected = (left + pad_x, top + pad_y)
            if overlap_penalty == 0:
                break
    text_left, text_top = selected
    box = (text_left - pad_x, text_top - pad_y, text_left + tw + pad_x, text_top + th + pad_y)
    _panel(image, *box, alpha=0.76)
    _draw_text_centered(image, text, box, color, font_size)
    return box


def _label_next_to_target(
    image: np.ndarray,
    text: str,
    target: tuple[float, float],
    side: str,
    color: Color,
    scale: float = 0.44,
) -> None:
    font_size = max(11, int(round(scale * 28)))
    font = _ui_font(font_size)
    tw, th = _measure_text(text, font)
    marker_half = 22
    gap = 16
    target_x, target_y = target
    if side == "left":
        x = int(round(target_x)) - marker_half - gap - tw
    else:
        x = int(round(target_x)) + marker_half + gap
    y = int(round(target_y)) + th // 2
    _label(image, text, (x, y), color, scale=scale)


def _draw_dashed_line(
    image: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Color,
    thickness: int = 1,
    dash: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for start in range(0, length, dash * 2):
        end = min(start + dash, length)
        t1 = start / length
        t2 = end / length
        a = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        b = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(image: np.ndarray, roi: Sequence[int], color: Color, show_labels: bool) -> None:
    x1, y1, x2, y2 = [int(v) for v in roi]
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)


def _format_value(px_value: Optional[float], settings: MeasurementSettings) -> str:
    if px_value is None:
        return "-"
    value = px_value * settings.calibration.px_to_real
    return f"{value:.2f} {settings.calibration.unit}"


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _taper_target_y(y_min: float, y_max: float, side: str, settings: MeasurementSettings) -> float:
    base_pct = float(getattr(settings, "base_height_pct", 30.0))
    offset_pct = float(
        getattr(settings, "right_offset_pct", 0.0)
        if side == "right"
        else getattr(settings, "left_offset_pct", 0.0)
    )
    target_pct = _clamp_pct(base_pct + offset_pct)
    return float(y_max) - (float(y_max - y_min) * target_pct / 100.0)


def _taper_sides(settings: MeasurementSettings) -> tuple[str, ...]:
    if settings.measurement_type == "taper_double":
        return ("left", "right")
    if settings.measurement_type == "taper_single":
        return (settings.taper_side if settings.taper_side in ("left", "right") else "left",)
    return ()


def _taper_for_side(result: Optional[MeasurementResult], side: str) -> Optional[TaperSideResult]:
    if result is None:
        return None
    return result.right_taper if side == "right" else result.left_taper


def _taper_fit_target_point(taper: TaperSideResult, settings: MeasurementSettings) -> Optional[tuple[float, float]]:
    if not taper.fit_line:
        return None
    x1, y1, x2, y2 = taper.fit_line
    y_min, y_max = min(float(y1), float(y2)), max(float(y1), float(y2))
    if y_max - y_min <= 1e-6:
        return float((x1 + x2) / 2.0), float(y_min)
    target_y = _taper_target_y(y_min, y_max, taper.side, settings)
    ratio = (target_y - float(y1)) / (float(y2) - float(y1))
    target_x = float(x1) + (float(x2) - float(x1)) * ratio
    return target_x, target_y


def _line_x_at_y(line, target_y: Optional[float]) -> Optional[tuple[float, float]]:
    if not line or target_y is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in line]
    if abs(y2 - y1) <= 1e-6:
        return (x1 + x2) * 0.5, float(target_y)
    ratio = (float(target_y) - y1) / (y2 - y1)
    return x1 + (x2 - x1) * ratio, float(target_y)


def _draw_target_marker(image: np.ndarray, target: tuple[float, float], color: Color, radius: int = 22) -> Tuple[int, int, int, int]:
    cx, cy = int(round(target[0])), int(round(target[1]))
    _stroke_line(image, (cx - radius, cy), (cx + radius, cy), color, 2, 4)
    cv2.circle(image, (cx, cy), 4, SHADOW, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 3, color, -1, cv2.LINE_AA)
    return (cx - radius - 6, cy - 8, cx + radius + 6, cy + 8)


def _draw_taper_height_guides(
    image: np.ndarray,
    roi: Sequence[int],
    settings: MeasurementSettings,
    result: Optional[MeasurementResult],
) -> None:
    sides = _taper_sides(settings)
    if not sides:
        return
    x1, y1, x2, y2 = [int(v) for v in roi]
    for side in sides:
        taper = _taper_for_side(result, side)
        if taper is None:
            continue
        target = _taper_fit_target_point(taper, settings)
        if target is None:
            continue
        target_x, target_y_float = target
        target_y = int(round(target_y_float))
        guide_half = int(max(18, min(30, abs(x2 - x1) * 0.055)))
        guide_x1 = max(x1, int(round(target_x)) - guide_half)
        guide_x2 = min(x2, int(round(target_x)) + guide_half)
        _stroke_line(image, (guide_x1, target_y), (guide_x2, target_y), TAPER_HEIGHT_COLOR, 2, 4)


def _draw_distance(image: np.ndarray, result: DistanceResult, settings: MeasurementSettings, color: Color, label_prefix: str) -> None:
    if not result.selected_pairs:
        return

    first_points = []
    second_points = []
    for pair in result.selected_pairs:
        if result.orientation == "horizontal":
            first_points.append((int(round(pair.first.image_x)), int(round(pair.first.image_y))))
            second_points.append((int(round(pair.second.image_x)), int(round(pair.second.image_y))))
        else:
            first_points.append((int(round(pair.first.image_x)), int(round(pair.first.image_y))))
            second_points.append((int(round(pair.second.image_x)), int(round(pair.second.image_y))))

    if len(first_points) >= 2:
        _stroke_polyline(image, np.asarray(first_points, dtype=np.int32), False, color, 2)
    if len(second_points) >= 2:
        _stroke_polyline(image, np.asarray(second_points, dtype=np.int32), False, color, 2)

    pair = result.selected_pair or result.selected_pairs[len(result.selected_pairs) // 2]
    p1 = (int(round(pair.first.image_x)), int(round(pair.first.image_y)))
    p2 = (int(round(pair.second.image_x)), int(round(pair.second.image_y)))
    _stroke_line(image, p1, p2, color, 2, 5)
    cv2.circle(image, p1, 4, color, -1, cv2.LINE_AA)
    cv2.circle(image, p2, 4, color, -1, cv2.LINE_AA)
    if settings.show_labels:
        mid = (int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2)))
        _label(
            image,
            f"{label_prefix} {_format_value(result.selected_px, settings)}",
            (mid[0] + 8, mid[1] - 8),
            color,
            avoid=(_bbox_for_points([p1, p2], 14),),
        )


def _draw_taper(image: np.ndarray, taper: TaperSideResult, color: Color, settings: MeasurementSettings, language: str) -> None:
    if taper.points and settings.show_selected_edges:
        _stroke_trace_segments(image, taper.points, color, 1, min_points=10, longest_only=True)
    if settings.show_fit_line and taper.fit_line:
        x1, y1, x2, y2 = taper.fit_line
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        _stroke_line_alpha(image, p1, p2, color, 1, 0.45)
        if settings.show_labels:
            angle = taper.angle_horizontal if taper.angle_horizontal is not None else 0.0
            side = taper_side_label(language, taper.side)
            target = _taper_fit_target_point(taper, settings)
            text = f"{side} {t(language, 'taper')} {angle:.1f} deg"
            if target is not None:
                _label_next_to_target(image, text, target, taper.side, color, scale=0.44)
            else:
                label_x = int(round(max(x1, x2) + 12 if taper.side == "left" else min(x1, x2) - 92))
                label_y = int(round(min(y1, y2) + 28))
                _label(image, text, (label_x, label_y), color, scale=0.44)


def _draw_ellipse_cd(image: np.ndarray, result: EllipseCDResult, settings: MeasurementSettings, language: str) -> None:
    if result.center_x is None or result.center_y is None or result.major_axis_px is None or result.minor_axis_px is None:
        return
    center = (int(round(result.center_x)), int(round(result.center_y)))
    axes = (max(1, int(round(result.major_axis_px * 0.5))), max(1, int(round(result.minor_axis_px * 0.5))))
    angle = float(result.angle_deg or 0.0)
    for x, y in result.boundary_points:
        point = (int(round(x)), int(round(y)))
    cv2.ellipse(image, center, axes, angle, 0.0, 360.0, SHADOW, 5, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, angle, 0.0, 360.0, ELLIPSE_COLOR, 2, cv2.LINE_AA)
    if result.horizontal_diameter_px is not None:
        half = int(round(result.horizontal_diameter_px * 0.5))
        p1 = (center[0] - half, center[1])
        p2 = (center[0] + half, center[1])
        _stroke_line(image, p1, p2, CD_COLOR, 2, 5)
        if settings.show_labels:
            _label(image, f"{t(language, 'ellipse_cd_horizontal')} {_format_value(result.horizontal_diameter_px, settings)}", (p2[0] + 8, p2[1] - 8), CD_COLOR, scale=0.44)
    if result.vertical_diameter_px is not None:
        half = int(round(result.vertical_diameter_px * 0.5))
        p1 = (center[0], center[1] - half)
        p2 = (center[0], center[1] + half)
        _stroke_line(image, p1, p2, THK_COLOR, 2, 5)
        if settings.show_labels:
            _label(image, f"{t(language, 'ellipse_cd_vertical')} {_format_value(result.vertical_diameter_px, settings)}", (p2[0] + 8, p2[1] - 8), THK_COLOR, scale=0.44)


def _draw_hole_cd(image: np.ndarray, result, settings: MeasurementSettings) -> None:
    if not result.contour_points:
        return
    contour = np.asarray(result.contour_points, dtype=np.int32).reshape(-1, 1, 2)
    _stroke_polyline(image, contour, True, ELLIPSE_COLOR, 2)
    points = np.asarray(result.contour_points, dtype=np.float32)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    cy = int(round((y_min + y_max) * 0.5))
    cx = int(round((x_min + x_max) * 0.5))
    _stroke_line(image, (int(round(x_min)), cy), (int(round(x_max)), cy), CD_COLOR, 2, 5)
    _stroke_line(image, (cx, int(round(y_min))), (cx, int(round(y_max))), THK_COLOR, 2, 5)
    if settings.show_labels:
        label = f"H {_format_value(result.horizontal_px, settings)}\nV {_format_value(result.vertical_px, settings)}"
        avoid = (
            _bbox_for_points([(int(round(x_min)), cy), (int(round(x_max)), cy)], 14),
            _bbox_for_points([(cx, int(round(y_min))), (cx, int(round(y_max)))], 14),
            _bbox_for_points([(int(round(x_min)), int(round(y_min))), (int(round(x_max)), int(round(y_max)))], 6),
        )
        _label(image, label, (int(round(x_max)) + 8, cy - 8), ELLIPSE_COLOR, scale=0.46, avoid=avoid)


def _draw_line_tuple(image: np.ndarray, line, color: Color, thickness: int = 2, alpha: Optional[float] = None) -> None:
    if not line:
        return
    x1, y1, x2, y2 = line
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    if alpha is None:
        _stroke_line(image, p1, p2, color, thickness, thickness + 3)
    else:
        _stroke_line_alpha(image, p1, p2, color, thickness, alpha)


def _draw_crater(image: np.ndarray, result, settings: MeasurementSettings) -> None:
    avoid_boxes: list[Tuple[int, int, int, int]] = []
    if result.top_profile_points and result.baseline_line:
        top_points = sorted(((float(x), float(y)) for x, y in result.top_profile_points), key=lambda item: item[0])
        bx1, by1, bx2, by2 = result.baseline_line
        denom = float(bx2 - bx1)
        if abs(denom) > 1e-6 and len(top_points) >= 3:
            bottom_points = [
                (
                    int(round(x)),
                    int(round(float(by1) + (float(by2) - float(by1)) * ((float(x) - float(bx1)) / denom))),
                )
                for x, _y in reversed(top_points)
            ]
            polygon = np.asarray(
                [(int(round(x)), int(round(y))) for x, y in top_points] + bottom_points,
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon], ELLIPSE_COLOR, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.14, image, 0.86, 0, image)
    if result.top_profile_points:
        points = np.asarray(result.top_profile_points, dtype=np.int32).reshape(-1, 1, 2)
        _stroke_polyline(image, points, False, ELLIPSE_COLOR, 2)
    _draw_line_tuple(image, result.baseline_line, THK_COLOR, 2)
    _draw_line_tuple(image, result.cd_line, CD_COLOR, 2)
    _draw_line_tuple(image, result.thk_line, THK_COLOR, 2)
    if settings.show_fit_line:
        _draw_line_tuple(image, result.left_taper_line, TAPER_LEFT_COLOR, 1, 0.45)
        _draw_line_tuple(image, result.right_taper_line, TAPER_RIGHT_COLOR, 1, 0.45)
    left_target = _line_x_at_y(result.left_taper_line, result.left_taper_measure_y)
    right_target = _line_x_at_y(result.right_taper_line, result.right_taper_measure_y)
    if left_target is not None:
        avoid_boxes.append(_draw_target_marker(image, left_target, TAPER_LEFT_COLOR))
    if right_target is not None:
        avoid_boxes.append(_draw_target_marker(image, right_target, TAPER_RIGHT_COLOR))
    for point in ((result.left_foot_x, result.left_foot_y), (result.right_foot_x, result.right_foot_y)):
        if point[0] is not None and point[1] is not None:
            cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 4, CD_COLOR, -1, cv2.LINE_AA)
    if settings.show_labels:
        if result.cd_line and result.cd_px is not None:
            x1, y1, x2, y2 = result.cd_line
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            _label(
                image,
                f"Crater CD {_format_value(result.cd_px, settings)}",
                (int(max(x1, x2)) + 10, int((y1 + y2) * 0.5) - 10),
                CD_COLOR,
                scale=0.46,
                avoid=tuple(avoid_boxes + [_bbox_for_points([p1, p2], 14)]),
            )
        if result.thk_line and result.thk_px is not None:
            x1, y1, x2, y2 = result.thk_line
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            _label(
                image,
                f"THK {_format_value(result.thk_px, settings)}",
                (int(max(x1, x2)) + 8, int((y1 + y2) * 0.5)),
                THK_COLOR,
                scale=0.46,
                avoid=tuple(avoid_boxes + [_bbox_for_points([p1, p2], 14)]),
            )
        if result.left_taper_line and result.left_taper_angle_horizontal is not None:
            x1, y1, x2, y2 = result.left_taper_line
            _label(
                image,
                f"L {result.left_taper_angle_horizontal:.1f} deg",
                (int(min(x1, x2)) - 10, int(min(y1, y2)) - 8),
                TAPER_LEFT_COLOR,
                scale=0.42,
                avoid=tuple(avoid_boxes),
            )
        if result.right_taper_line and result.right_taper_angle_horizontal is not None:
            x1, y1, x2, y2 = result.right_taper_line
            _label(
                image,
                f"R {result.right_taper_angle_horizontal:.1f} deg",
                (int(max(x1, x2)) + 8, int(min(y1, y2)) - 8),
                TAPER_RIGHT_COLOR,
                scale=0.42,
                avoid=tuple(avoid_boxes),
            )
        if result.warning_message and any(
            code in result.warning_message
            for code in (
                "crater_baseline_coverage_low",
                "crater_baseline_unstable",
                "crater_roi_needs_more_baseline",
                "crater_baseline_may_be_too_high",
                "crater_partial_dome_detected",
            )
        ):
            anchor = (14, int(max(38, min(image.shape[0] - 36, (result.baseline_line or (0, image.shape[0] * 0.65, 0, 0))[1] - 36))))
            _label(
                image,
                "Check baseline / widen ROI",
                anchor,
                FAIL_COLOR,
                scale=0.42,
                avoid=tuple(avoid_boxes),
            )


def _draw_legend(image: np.ndarray, result: Optional[MeasurementResult], settings: MeasurementSettings, language: str) -> None:
    return


def _draw_summary(image: np.ndarray, result: MeasurementResult, settings: MeasurementSettings, language: str) -> None:
    if not settings.show_labels:
        return
    status_color = THK_COLOR if result.status == "OK" else ROI_COLOR if result.status == "Check" else FAIL_COLOR
    display_status = status_label(language, result.status)
    lines = [
        f"{display_status}  {result.overall_confidence:.0f}%",
        measurement_label(language, settings.measurement_type),
    ]
    has_composite_result = result.hole_cd is not None or result.crater is not None
    if not has_composite_result and result.horizontal_cd and result.horizontal_cd.selected_px is not None:
        lines.append(f"CD {_format_value(result.horizontal_cd.selected_px, settings)}")
    if not has_composite_result and result.vertical_thk and result.vertical_thk.selected_px is not None:
        lines.append(f"THK {_format_value(result.vertical_thk.selected_px, settings)}")
    if result.avg_taper_angle is not None:
        lines.append(f"{t(language, 'average_taper')} {result.avg_taper_angle:.1f} deg")
    if result.ellipse_cd and result.ellipse_cd.horizontal_diameter_px is not None:
        lines.append(f"{t(language, 'ellipse_cd_horizontal')} {_format_value(result.ellipse_cd.horizontal_diameter_px, settings)}")
    if result.ellipse_cd and result.ellipse_cd.vertical_diameter_px is not None:
        lines.append(f"{t(language, 'ellipse_cd_vertical')} {_format_value(result.ellipse_cd.vertical_diameter_px, settings)}")
    if result.crater and result.crater.cd_px is not None:
        lines.append(f"Crater CD {_format_value(result.crater.cd_px, settings)}")
        if result.crater.thk_px is not None:
            lines.append(f"Crater THK {_format_value(result.crater.thk_px, settings)}")
        if result.crater.avg_taper_angle is not None:
            lines.append(f"{t(language, 'average_taper')} {result.crater.avg_taper_angle:.1f} deg")
        if result.crater.warning_message and any(
            code in result.crater.warning_message
            for code in ("crater_roi_needs_more_baseline", "crater_partial_dome_detected", "crater_baseline_may_be_too_high")
        ):
            lines.append("Check baseline / widen ROI")

    x = 14
    y = 14
    font_size = 13
    font = _ui_font(font_size)
    width = max(190, min(360, max(_measure_text(line, font)[0] for line in lines) + 28))
    height = 16 + len(lines) * 20
    _panel(image, x, y, x + width, y + height, alpha=0.72)
    for idx, line in enumerate(lines):
        line_top = y + 6 + idx * 20
        _draw_text_centered(
            image,
            line,
            (x + 8, line_top, x + width - 8, line_top + 20),
            status_color if idx == 0 else TEXT,
            font_size,
        )


def draw_overlay(
    image: np.ndarray,
    roi: Optional[Sequence[int]],
    result: Optional[MeasurementResult],
    settings: MeasurementSettings,
    show_overlay: bool = True,
    scale_bar_bbox: Optional[Tuple[int, int, int, int]] = None,
    language: str = "ko",
) -> np.ndarray:
    canvas = image.copy()
    if not show_overlay:
        return canvas

    if roi is not None and settings.show_roi:
        _draw_dashed_rect(canvas, roi, ROI_COLOR, settings.show_labels)
    if roi is not None:
        _draw_taper_height_guides(canvas, roi, settings, result)
    if scale_bar_bbox:
        x1, y1, x2, y2 = [int(v) for v in scale_bar_bbox]
        if abs(y2 - y1) <= 2:
            _stroke_line(canvas, (x1, y1), (max(x1, x2 - 1), y1), (255, 255, 90), 2, 4)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 90), 2, cv2.LINE_AA)
        if settings.show_labels:
            _label(canvas, t(language, "scale_bar"), (x1, y1 - 8), (255, 255, 90), scale=0.48)
    if result is not None:
        if settings.show_selected_edges:
            has_composite_result = result.hole_cd is not None or result.crater is not None
            if result.horizontal_cd and not has_composite_result:
                _draw_distance(canvas, result.horizontal_cd, settings, CD_COLOR, "CD")
            if result.vertical_thk and not has_composite_result:
                _draw_distance(canvas, result.vertical_thk, settings, THK_COLOR, "THK")
            if result.ellipse_cd:
                _draw_ellipse_cd(canvas, result.ellipse_cd, settings, language)
            if result.hole_cd:
                _draw_hole_cd(canvas, result.hole_cd, settings)
            if result.crater:
                _draw_crater(canvas, result.crater, settings)
        if result.left_taper:
            _draw_taper(canvas, result.left_taper, TAPER_LEFT_COLOR, settings, language)
        if result.right_taper:
            _draw_taper(canvas, result.right_taper, TAPER_RIGHT_COLOR, settings, language)
        _draw_summary(canvas, result, settings, language)
    _draw_legend(canvas, result, settings, language)
    return canvas
