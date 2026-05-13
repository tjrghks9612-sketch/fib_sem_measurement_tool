from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from fib_sem_measurement_tool.models.settings import CalibrationSettings


def detect_scale_bar(image: np.ndarray) -> Dict[str, object]:
    """Detect horizontal pure-green scale bar near the lower-left area.

    Priority is given to connected components within the lower-left search window,
    while still allowing fallback detection on the full image.
    """
    if image is None or image.size == 0:
        return {"status": "not_found", "pixel_length": None, "bbox": None, "message": "이미지가 비어 있습니다"}

    # BGR image from OpenCV: green line is (0, 255, 0) in RGB/BGR with a small tolerance.
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    green_mask = (g >= 245) & (b <= 12) & (r <= 12)

    height, width = image.shape[:2]
    if not np.any(green_mask):
        return {
            "status": "not_found",
            "pixel_length": None,
            "bbox": None,
            "message": "좌하단의 초록색 스케일바(0,255,0)를 찾지 못했습니다",
        }

    search_windows = [
        ("bottom_left", 0, int(width * 0.55), int(height * 0.60), height),
        ("full", 0, width, 0, height),
    ]

    best: Optional[Dict[str, object]] = None
    for region_name, x0, x1, y0, y1 in search_windows:
        crop = green_mask[y0:y1, x0:x1]
        if crop.size == 0 or not np.any(crop):
            continue

        binary = (crop.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < max(12, int(width * 0.03)):
                continue
            if h > max(12, int(height * 0.05)):
                continue
            aspect = w / max(h, 1)
            if aspect < 4.0:
                continue
            score = float(w) * (1.2 if region_name == "bottom_left" else 1.0)
            candidate = {
                "status": "detected",
                "region": region_name,
                "pixel_length": float(w),
                "bbox": (int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h)),
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return {
            "status": "not_found",
            "pixel_length": None,
            "bbox": None,
            "message": "초록색 스케일바 후보를 찾지 못했습니다",
        }
    return best


def apply_calibration(
    pixel_length: float,
    actual_length: float,
    unit: str,
    mode: str = "auto",
) -> CalibrationSettings:
    if pixel_length <= 0 or actual_length <= 0:
        return CalibrationSettings(status="failed", mode=mode, unit=unit)
    px_to_real = float(actual_length) / float(pixel_length)
    return CalibrationSettings(
        px_to_real=px_to_real,
        unit=unit,
        mode=mode,
        detected_scale_bar_px=float(pixel_length) if mode == "auto" else None,
        actual_scale_bar_length=float(actual_length),
        status="calibrated",
        manual_pixel_length=float(pixel_length) if mode == "manual" else None,
    )


def clone_calibration(calibration: CalibrationSettings) -> CalibrationSettings:
    return replace(calibration)

