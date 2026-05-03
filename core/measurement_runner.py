from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from fib_sem_measurement_tool.core.image_io import to_gray
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_taper import measure_double_taper, measure_single_taper
from fib_sem_measurement_tool.core.preprocessing import preprocess_image
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import MeasurementResult, MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


logger = logging.getLogger(__name__)


class MeasurementRoiError(ValueError):
    pass


def _status_by_threshold(confidence: float, threshold: float, failed: bool = False) -> str:
    if failed:
        return MeasurementStatus.FAIL.value
    check_cut = max(0.0, threshold - 20.0)
    if confidence >= threshold:
        return MeasurementStatus.OK.value
    if confidence >= check_cut:
        return MeasurementStatus.CHECK.value
    return MeasurementStatus.REVIEW_NEEDED.value


def _fail_result(measurement_type: str, message: str) -> MeasurementResult:
    return MeasurementResult(measurement_type=measurement_type, overall_confidence=0.0, status=MeasurementStatus.FAIL.value, warning_message=message)


def calculate_confidence(result: MeasurementResult, settings: MeasurementSettings) -> float:
    scores = [item.confidence for item in (result.horizontal_cd, result.vertical_thk, result.left_taper, result.right_taper) if item is not None and item.status != MeasurementStatus.FAIL.value]
    if not scores:
        result.overall_confidence = 0.0
        result.status = MeasurementStatus.FAIL.value
        return 0.0
    result.overall_confidence = float(np.mean(scores))
    result.status = _status_by_threshold(result.overall_confidence, settings.advanced.confidence_threshold)
    return result.overall_confidence


def run_measurement(image: np.ndarray, settings: MeasurementSettings) -> MeasurementResult:
    try:
        gray = to_gray(image)
        processed_gray = preprocess_image(gray, settings)
        roi = settings.roi
        if roi is None:
            raise MeasurementRoiError("ROI가 지정되지 않았습니다")
        clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
        if clean_roi is None:
            raise MeasurementRoiError("ROI가 없거나 너무 작습니다")

        if settings.measurement_type == "distance_horizontal":
            result = MeasurementResult(measurement_type=settings.measurement_type, horizontal_cd=measure_horizontal_cd(processed_gray, clean_roi, settings))
        elif settings.measurement_type == "distance_vertical":
            result = MeasurementResult(measurement_type=settings.measurement_type, vertical_thk=measure_vertical_thk(processed_gray, clean_roi, settings))
        elif settings.measurement_type == "distance_both":
            horizontal = measure_horizontal_cd(processed_gray, clean_roi, settings)
            vertical = measure_vertical_thk(processed_gray, clean_roi, settings)
            result = MeasurementResult(measurement_type=settings.measurement_type, horizontal_cd=horizontal, vertical_thk=vertical, warning_message="; ".join([i.warning_message for i in (horizontal, vertical) if i.warning_message]))
        elif settings.measurement_type == "taper_double":
            result = measure_double_taper(processed_gray, clean_roi, settings)
            result.measurement_type = settings.measurement_type
        elif settings.measurement_type == "taper_single":
            result = measure_single_taper(processed_gray, clean_roi, settings.taper_side, settings)
            result.measurement_type = settings.measurement_type
        else:
            return _fail_result(settings.measurement_type, "지원하지 않는 측정 타입입니다")
        calculate_confidence(result, settings)
        return result
    except MeasurementRoiError as exc:
        logger.info(
            "measurement_failed type=roi_error measurement_type=%s roi=%s",
            settings.measurement_type,
            settings.roi,
        )
        return _fail_result(settings.measurement_type, str(exc))
    except (TypeError, ValueError, FloatingPointError) as exc:
        logger.exception(
            "measurement_failed type=calculation_error measurement_type=%s roi=%s",
            settings.measurement_type,
            settings.roi,
        )
        return _fail_result(settings.measurement_type, f"측정 연산 오류: {exc}")
    except Exception as exc:
        logger.exception(
            "measurement_failed type=unknown_error measurement_type=%s roi=%s",
            settings.measurement_type,
            settings.roi,
        )
        return _fail_result(settings.measurement_type, f"측정 중 오류: {exc}")
