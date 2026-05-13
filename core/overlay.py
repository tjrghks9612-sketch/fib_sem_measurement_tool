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

BG_PANEL: Color = (10, 18, 28)
TEXT: Color = (232, 240, 248)
MUTED: Color = (150, 165, 180)
ROI_COLOR: Color = (0, 210, 255)
CD_COLOR: Color = (255, 164, 55)
THK_COLOR: Color = (80, 245, 120)
TAPER_LEFT_COLOR: Color = (255, 220, 75)
TAPER_RIGHT_COLOR: Color = (255, 145, 85)
TAPER_HEIGHT_COLOR: Color = (210, 190, 255)
ELLIPSE_COLOR: Color = (125, 205, 255)
POINT_COLOR: Color = (245, 250, 255)
FAIL_COLOR: Color = (80, 105, 255)


@lru_cache(maxsize=12)
def _ui_font(size: int):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
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


def _label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Color, scale: float = 0.55) -> None:
    font_size = max(11, int(round(scale * 28)))
    font = _ui_font(font_size)
    x, y = origin
    tw, th = _measure_text(text, font)
    x = max(4, min(int(x), image.shape[1] - tw - 10))
    y = max(th + 10, min(int(y), image.shape[0] - 8))
    text_top = y - th
    pad_x = 7
    pad_y = 6
    overlay = image.copy()
    cv2.rectangle(overlay, (x - pad_x, text_top - pad_y), (x + tw + pad_x, text_top + th + pad_y), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    _draw_text(image, text, (x, text_top), color, font_size)


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
    _draw_dashed_line(image, (x1, y1), (x2, y1), color, 1)
    _draw_dashed_line(image, (x2, y1), (x2, y2), color, 1)
    _draw_dashed_line(image, (x2, y2), (x1, y2), color, 1)
    _draw_dashed_line(image, (x1, y2), (x1, y1), color, 1)
    if show_labels:
        _label(image, "ROI", (x1 + 8, y1 + 22), ROI_COLOR)


def _format_value(px_value: Optional[float], settings: MeasurementSettings) -> str:
    if px_value is None:
        return "-"
    value = px_value * settings.calibration.px_to_real
    return f"{value:.3g} {settings.calibration.unit}"


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
        cv2.line(image, (guide_x1, target_y), (guide_x2, target_y), (20, 20, 26), 5, cv2.LINE_AA)
        cv2.line(image, (guide_x1, target_y), (guide_x2, target_y), TAPER_HEIGHT_COLOR, 2, cv2.LINE_AA)


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

    for point in first_points + second_points:
        cv2.circle(image, point, 2, POINT_COLOR, -1, cv2.LINE_AA)
    if len(first_points) >= 2:
        cv2.polylines(image, [np.asarray(first_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    if len(second_points) >= 2:
        cv2.polylines(image, [np.asarray(second_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    pair = result.selected_pair or result.selected_pairs[len(result.selected_pairs) // 2]
    p1 = (int(round(pair.first.image_x)), int(round(pair.first.image_y)))
    p2 = (int(round(pair.second.image_x)), int(round(pair.second.image_y)))
    cv2.arrowedLine(image, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.025)
    cv2.arrowedLine(image, p2, p1, color, 2, cv2.LINE_AA, tipLength=0.025)
    if settings.show_labels:
        mid = (int(round((p1[0] + p2[0]) / 2)), int(round((p1[1] + p2[1]) / 2)))
        _label(image, f"{label_prefix} {_format_value(result.selected_px, settings)}", (mid[0] + 8, mid[1] - 8), color)


def _draw_taper(image: np.ndarray, taper: TaperSideResult, color: Color, settings: MeasurementSettings, language: str) -> None:
    if settings.show_selected_edges:
        for candidate in taper.selected_boundary_candidates[:: max(1, len(taper.selected_boundary_candidates) // 36)]:
            point = (int(round(candidate.image_x)), int(round(candidate.image_y)))
            cv2.circle(image, point, 1, POINT_COLOR, -1, cv2.LINE_AA)
    if settings.show_fit_line and taper.fit_line:
        x1, y1, x2, y2 = taper.fit_line
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(image, p1, p2, (20, 20, 26), 4, cv2.LINE_AA)
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
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
        cv2.circle(image, point, 3, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(image, point, 4, ELLIPSE_COLOR, 1, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, angle, 0.0, 360.0, (20, 20, 26), 5, cv2.LINE_AA)
    cv2.ellipse(image, center, axes, angle, 0.0, 360.0, ELLIPSE_COLOR, 2, cv2.LINE_AA)
    if result.horizontal_diameter_px is not None:
        half = int(round(result.horizontal_diameter_px * 0.5))
        p1 = (center[0] - half, center[1])
        p2 = (center[0] + half, center[1])
        cv2.arrowedLine(image, p1, p2, CD_COLOR, 2, cv2.LINE_AA, tipLength=0.025)
        cv2.arrowedLine(image, p2, p1, CD_COLOR, 2, cv2.LINE_AA, tipLength=0.025)
        if settings.show_labels:
            _label(image, f"{t(language, 'ellipse_cd_horizontal')} {_format_value(result.horizontal_diameter_px, settings)}", (p2[0] + 8, p2[1] - 8), CD_COLOR, scale=0.44)
    if result.vertical_diameter_px is not None:
        half = int(round(result.vertical_diameter_px * 0.5))
        p1 = (center[0], center[1] - half)
        p2 = (center[0], center[1] + half)
        cv2.arrowedLine(image, p1, p2, THK_COLOR, 2, cv2.LINE_AA, tipLength=0.025)
        cv2.arrowedLine(image, p2, p1, THK_COLOR, 2, cv2.LINE_AA, tipLength=0.025)
        if settings.show_labels:
            _label(image, f"{t(language, 'ellipse_cd_vertical')} {_format_value(result.vertical_diameter_px, settings)}", (p2[0] + 8, p2[1] - 8), THK_COLOR, scale=0.44)


def _draw_legend(image: np.ndarray, result: Optional[MeasurementResult], settings: MeasurementSettings, language: str) -> None:
    if not settings.show_labels:
        return
    rows = [("ROI", ROI_COLOR)]
    if settings.show_selected_edges:
        rows.append((t(language, "selected_edges"), POINT_COLOR))
    if settings.show_fit_line and result and (result.left_taper or result.right_taper):
        rows.append((t(language, "fit_line"), TAPER_LEFT_COLOR))
    if result and result.ellipse_cd:
        rows.append((measurement_label(language, "ellipse_cd"), ELLIPSE_COLOR))
    if settings.roi is not None and _taper_sides(settings):
        rows.append((t(language, "taper_height"), TAPER_HEIGHT_COLOR))

    x = max(12, image.shape[1] - 220)
    y = 16
    width = 198
    height = 18 + 22 * len(rows)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    for idx, (label, color) in enumerate(rows):
        yy = y + 16 + idx * 22
        cv2.line(image, (x + 12, yy), (x + 30, yy), color, 2, cv2.LINE_AA)
        _draw_text(image, label, (x + 40, yy - 8), TEXT, 12)


def _draw_summary(image: np.ndarray, result: MeasurementResult, settings: MeasurementSettings, language: str) -> None:
    if not settings.show_labels:
        return
    status_color = THK_COLOR if result.status == "OK" else ROI_COLOR if result.status == "Check" else FAIL_COLOR
    display_status = status_label(language, result.status)
    lines = [
        f"{display_status} / {t(language, 'confidence')} {result.overall_confidence:.0f}%",
        measurement_label(language, settings.measurement_type),
        f"{t(language, 'raw_edge_count')} {result.raw_edge_count()} / {t(language, 'selected_points')} {result.selected_point_count()}",
    ]
    if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
        lines.append(f"CD {_format_value(result.horizontal_cd.selected_px, settings)}")
    if result.vertical_thk and result.vertical_thk.selected_px is not None:
        lines.append(f"THK {_format_value(result.vertical_thk.selected_px, settings)}")
    if result.avg_taper_angle is not None:
        lines.append(f"{t(language, 'average_taper')} {result.avg_taper_angle:.1f} deg")
    if result.ellipse_cd and result.ellipse_cd.horizontal_diameter_px is not None:
        lines.append(f"{t(language, 'ellipse_cd_horizontal')} {_format_value(result.ellipse_cd.horizontal_diameter_px, settings)}")
    if result.ellipse_cd and result.ellipse_cd.vertical_diameter_px is not None:
        lines.append(f"{t(language, 'ellipse_cd_vertical')} {_format_value(result.ellipse_cd.vertical_diameter_px, settings)}")

    x = 16
    y = 20
    font_size = 14
    font = _ui_font(font_size)
    width = max(270, min(440, max(_measure_text(line, font)[0] for line in lines) + 30))
    height = 20 + len(lines) * 22
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.74, image, 0.26, 0, image)
    for idx, line in enumerate(lines):
        _draw_text(image, line, (x + 12, y + 12 + idx * 22), status_color if idx == 0 else TEXT, font_size)


def draw_overlay(
    image: np.ndarray,
    roi: Optional[Sequence[int]],
    result: Optional[MeasurementResult],
    settings: MeasurementSettings,
    show_overlay: bool = True,
    language: str = "ko",
) -> np.ndarray:
    canvas = image.copy()
    if not show_overlay:
        return canvas

    if roi is not None and settings.show_roi:
        _draw_dashed_rect(canvas, roi, ROI_COLOR, settings.show_labels)
    if roi is not None:
        _draw_taper_height_guides(canvas, roi, settings, result)
    if result is not None:
        if settings.show_selected_edges:
            if result.horizontal_cd:
                _draw_distance(canvas, result.horizontal_cd, settings, CD_COLOR, "CD")
            if result.vertical_thk:
                _draw_distance(canvas, result.vertical_thk, settings, THK_COLOR, "THK")
            if result.ellipse_cd:
                _draw_ellipse_cd(canvas, result.ellipse_cd, settings, language)
        if result.left_taper:
            _draw_taper(canvas, result.left_taper, TAPER_LEFT_COLOR, settings, language)
        if result.right_taper:
            _draw_taper(canvas, result.right_taper, TAPER_RIGHT_COLOR, settings, language)
        _draw_summary(canvas, result, settings, language)
    _draw_legend(canvas, result, settings, language)
    return canvas
