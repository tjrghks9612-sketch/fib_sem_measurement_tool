from __future__ import annotations

import logging

import numpy as np

from fib_sem_measurement_tool.core.image_io import to_gray
from fib_sem_measurement_tool.core.hole_cd_measurement import measure_hole_cd
from fib_sem_measurement_tool.core.measurement_cd_thk import CDMeasurementEngine
from fib_sem_measurement_tool.core.measurement_taper import measure_double_taper, measure_single_taper
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import MeasurementResult, MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


logger = logging.getLogger(__name__)
COVERAGE_OK_PERCENT = 80.0


class MeasurementRoiError(ValueError):
    pass


class UnsupportedMeasurementTypeError(ValueError):
    pass


def _measurement_log_context(settings: MeasurementSettings, image: np.ndarray) -> dict:
    return {
        "measurement_type": settings.measurement_type,
        "roi": settings.roi,
        "minimum_grayscale_delta": settings.minimum_grayscale_delta,
        "image_shape": tuple(int(v) for v in image.shape),
    }


def _status_by_coverage(coverage_percent: float, threshold: float, failed: bool = False) -> str:
    if failed:
        return MeasurementStatus.FAIL.value
    check_cut = max(0.0, threshold - 20.0)
    if coverage_percent >= threshold:
        return MeasurementStatus.OK.value
    if coverage_percent >= check_cut:
        return MeasurementStatus.CHECK.value
    return MeasurementStatus.REVIEW_NEEDED.value


def _fail_result(measurement_type: str, message: str) -> MeasurementResult:
    return MeasurementResult(
        measurement_type=measurement_type,
        overall_confidence=0.0,
        status=MeasurementStatus.FAIL.value,
        warning_message=message,
    )


def calculate_overall_coverage(result: MeasurementResult) -> float:
    components = [
        item
        for item in (result.horizontal_cd, result.vertical_thk, result.left_taper, result.right_taper, result.ellipse_cd, result.hole_cd)
        if item is not None
    ]
    if not components:
        result.overall_confidence = 0.0
        result.status = MeasurementStatus.FAIL.value
        return 0.0

    coverage_scores = [
        0.0 if item.status == MeasurementStatus.FAIL.value else item.confidence
        for item in components
    ]
    result.overall_confidence = float(np.mean(coverage_scores))
    if all(item.status == MeasurementStatus.FAIL.value for item in components):
        result.status = MeasurementStatus.FAIL.value
    else:
        result.status = _status_by_coverage(result.overall_confidence, COVERAGE_OK_PERCENT)
    return result.overall_confidence

def run_measurement(image: np.ndarray, settings: MeasurementSettings) -> MeasurementResult:
    context = _measurement_log_context(settings, image)
    try:
        gray = to_gray(image)
        roi = settings.roi
        if roi is None:
            raise MeasurementRoiError("ROI is not set")
        clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
        if clean_roi is None:
            raise MeasurementRoiError("ROI is empty or too small")

        if settings.measurement_type in {"distance_horizontal", "distance_vertical", "distance_both"}:
            direction_by_type = {
                "distance_horizontal": "horizontal",
                "distance_vertical": "vertical",
                "distance_both": getattr(settings, "measure_direction", "both"),
            }
            result = CDMeasurementEngine(settings).measure(gray, clean_roi, direction_by_type[settings.measurement_type])
            result.measurement_type = settings.measurement_type
        elif settings.measurement_type == "taper_double":
            result = measure_double_taper(gray, clean_roi, settings)
            result.measurement_type = settings.measurement_type
        elif settings.measurement_type == "taper_single":
            result = measure_single_taper(gray, clean_roi, settings.taper_side, settings)
            result.measurement_type = settings.measurement_type
        elif settings.measurement_type == "hole_cd":
            result = measure_hole_cd(gray, clean_roi, settings)
            result.measurement_type = settings.measurement_type
        else:
            raise UnsupportedMeasurementTypeError("Unsupported measurement type")
        if result.hole_cd is None:
            calculate_overall_coverage(result)
        return result
    except MeasurementRoiError as exc:
        logger.info("measurement_failed type=roi_error message=%s context=%s", str(exc), context)
        return _fail_result(settings.measurement_type, str(exc))
    except UnsupportedMeasurementTypeError as exc:
        logger.warning("measurement_failed type=unsupported_measurement_type message=%s context=%s", str(exc), context)
        return _fail_result(settings.measurement_type, str(exc))
    except (TypeError, ValueError, FloatingPointError) as exc:
        logger.exception("measurement_failed type=calculation_error message=%s context=%s", str(exc), context)
        return _fail_result(settings.measurement_type, f"Measurement calculation error: {exc}")
    except Exception as exc:
        logger.exception("measurement_failed type=unknown_error message=%s context=%s", str(exc), context)
        return _fail_result(settings.measurement_type, f"Measurement error: {exc}")
