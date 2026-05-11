from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fib_sem_measurement_tool.models.result import (
    DistanceResult,
    MeasurementResult,
    RawEdgeCandidate,
    TaperSideResult,
)
from fib_sem_measurement_tool.models.settings import MEASUREMENT_TYPES, MeasurementSettings


Color = Tuple[int, int, int]

BG_PANEL: Color = (10, 18, 28)
TEXT: Color = (232, 240, 248)
MUTED: Color = (150, 165, 180)
ROI_COLOR: Color = (0, 210, 255)
RAW_HORIZONTAL_COLOR: Color = (80, 220, 255)
RAW_VERTICAL_COLOR: Color = (70, 245, 140)
CD_COLOR: Color = (255, 164, 55)
THK_COLOR: Color = (80, 245, 120)
TAPER_LEFT_COLOR: Color = (255, 220, 75)
TAPER_RIGHT_COLOR: Color = (255, 145, 85)
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
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 7, y - th - 9), (x + tw + 8, y + 7), BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    _draw_text(image, text, (x, y - th), color, font_size)


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


def _iter_raw_candidates(result: MeasurementResult) -> Iterable[RawEdgeCandidate]:
    seen: set[tuple[str, int, float, float, float]] = set()
    for measurement in (result.horizontal_cd, result.vertical_thk, result.left_taper, result.right_taper):
        if measurement is None:
            continue
        for candidate in measurement.raw_edge_candidates:
            key = (
                candidate.scan_axis,
                candidate.scan_index,
                round(candidate.position, 4),
                round(candidate.image_x, 4),
                round(candidate.image_y, 4),
            )
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _draw_raw_candidates(image: np.ndarray, candidates: Iterable[RawEdgeCandidate]) -> None:
    overlay = image.copy()
    count = 0
    for candidate in candidates:
        x = int(round(candidate.image_x))
        y = int(round(candidate.image_y))
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        if candidate.scan_axis == "horizontal":
            cv2.line(overlay, (x, max(0, y - 2)), (x, min(image.shape[0] - 1, y + 2)), RAW_HORIZONTAL_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.line(overlay, (max(0, x - 2), y), (min(image.shape[1] - 1, x + 2), y), RAW_VERTICAL_COLOR, 1, cv2.LINE_AA)
        count += 1
    if count:
        cv2.addWeighted(overlay, 0.36, image, 0.64, 0, image)


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


def _draw_taper(image: np.ndarray, taper: TaperSideResult, color: Color, settings: MeasurementSettings) -> None:
    if settings.show_selected_edges:
        for candidate in taper.selected_boundary_candidates:
            point = (int(round(candidate.image_x)), int(round(candidate.image_y)))
            cv2.circle(image, point, 3, POINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, point, 5, color, 1, cv2.LINE_AA)
    if settings.show_fit_line and taper.fit_line:
        x1, y1, x2, y2 = taper.fit_line
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(image, p1, p2, color, 1, cv2.LINE_AA)
        if settings.show_labels:
            angle = taper.angle_horizontal if taper.angle_horizontal is not None else 0.0
            mid = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
            side = "좌측" if taper.side == "left" else "우측"
            _label(image, f"{side} {angle:.1f} deg", (mid[0] + 8, mid[1]), color, scale=0.48)


def _draw_legend(image: np.ndarray, result: Optional[MeasurementResult], settings: MeasurementSettings) -> None:
    if not settings.show_labels:
        return
    rows = [("ROI", ROI_COLOR)]
    if settings.show_raw_candidates:
        rows.append(("원시 후보", RAW_HORIZONTAL_COLOR))
    if settings.show_selected_edges:
        rows.append(("선택 경계", POINT_COLOR))
    if settings.show_fit_line and result and (result.left_taper or result.right_taper):
        rows.append(("피팅 선", TAPER_LEFT_COLOR))

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


def _draw_summary(image: np.ndarray, result: MeasurementResult, settings: MeasurementSettings) -> None:
    if not settings.show_labels:
        return
    status_color = THK_COLOR if result.status == "OK" else ROI_COLOR if result.status == "Check" else FAIL_COLOR
    status_label = {
        "OK": "정상",
        "Check": "확인",
        "Review Needed": "검토 필요",
        "Fail": "실패",
    }.get(result.status, result.status)
    lines = [
        f"{status_label} / 신뢰도 {result.overall_confidence:.0f}%",
        MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type),
        f"원시 경계 {result.raw_edge_count()} / 선택 포인트 {result.selected_point_count()}",
    ]
    if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
        lines.append(f"CD {_format_value(result.horizontal_cd.selected_px, settings)}")
    if result.vertical_thk and result.vertical_thk.selected_px is not None:
        lines.append(f"THK {_format_value(result.vertical_thk.selected_px, settings)}")
    if result.avg_taper_angle is not None:
        lines.append(f"평균 {result.avg_taper_angle:.1f} deg")

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
    calibration_line: Optional[Tuple[int, int, int, int]] = None,
    scale_bar_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    canvas = image.copy()
    if not show_overlay:
        return canvas

    if roi is not None and settings.show_roi:
        _draw_dashed_rect(canvas, roi, ROI_COLOR, settings.show_labels)
    if scale_bar_bbox:
        x1, y1, x2, y2 = [int(v) for v in scale_bar_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 90), 2, cv2.LINE_AA)
        if settings.show_labels:
            _label(canvas, "스케일바", (x1, y1 - 8), (255, 255, 90), scale=0.48)
    if calibration_line:
        x1, y1, x2, y2 = [int(v) for v in calibration_line]
        cv2.line(canvas, (x1, y1), (x2, y2), (70, 220, 255), 2, cv2.LINE_AA)
        if settings.show_labels:
            _label(canvas, "캘리브레이션 선", (x1, y1 - 8), (70, 220, 255), scale=0.48)

    if result is not None:
        if settings.show_raw_candidates:
            _draw_raw_candidates(canvas, _iter_raw_candidates(result))
        if settings.show_selected_edges:
            if result.horizontal_cd:
                _draw_distance(canvas, result.horizontal_cd, settings, CD_COLOR, "CD")
            if result.vertical_thk:
                _draw_distance(canvas, result.vertical_thk, settings, THK_COLOR, "THK")
        if result.left_taper:
            _draw_taper(canvas, result.left_taper, TAPER_LEFT_COLOR, settings)
        if result.right_taper:
            _draw_taper(canvas, result.right_taper, TAPER_RIGHT_COLOR, settings)
        _draw_summary(canvas, result, settings)
    _draw_legend(canvas, result, settings)
    return canvas
