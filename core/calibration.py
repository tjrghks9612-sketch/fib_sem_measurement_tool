from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from fib_sem_measurement_tool.models.settings import CalibrationSettings


def detect_scale_bar(image: np.ndarray) -> Dict[str, object]:
    """Detect a horizontal green scale bar in the lower-left quadrant only."""
    if image is None or image.size == 0 or image.ndim < 3 or image.shape[2] < 3:
        return {"status": "not_found", "pixel_length": None, "bbox": None, "message": "scale_bar_not_found"}

    height, width = image.shape[:2]
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    green_mask = (g >= 200) & (r <= 80) & (b <= 80)

    x0, x1 = 0, width // 2
    y0, y1 = height // 2, height
    crop = green_mask[y0:y1, x0:x1]
    best: Dict[str, object] | None = None
    for local_y in range(crop.shape[0]):
        row = crop[local_y]
        run_start = None
        for idx, is_green in enumerate(np.r_[row, False]):
            if bool(is_green) and run_start is None:
                run_start = idx
            elif not bool(is_green) and run_start is not None:
                run_end = idx - 1
                length = run_end - run_start + 1
                if length >= 15 and (best is None or length > best["pixel_length"]):
                    y = y0 + local_y
                    best = {
                        "status": "detected",
                        "region": "lower_left_quadrant",
                        "pixel_length": float(length),
                        "bbox": (int(x0 + run_start), int(y), int(x0 + run_end + 1), int(y + 1)),
                        "line": (int(x0 + run_start), int(y), int(x0 + run_end), int(y)),
                        "score": float(length),
                    }
                run_start = None

    if best is None:
        return {"status": "not_found", "pixel_length": None, "bbox": None, "message": "scale_bar_not_found"}
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
