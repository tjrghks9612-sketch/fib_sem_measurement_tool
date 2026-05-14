from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Optional

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.result import DistanceResult, MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings, resolve_effective_settings


CSV_COLUMNS = [
    "file_name",
    "full_path",
    "measurement_type",
    "roi_source_image",
    "settings_source",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "distance_method",
    "minimum_grayscale_delta",
    "px_to_real",
    "unit",
    "calibration_mode",
    "calibration_status",
    "detected_scale_bar_px",
    "actual_scale_bar_length",
    "taper_side",
    "left_taper_angle_horizontal",
    "right_taper_angle_horizontal",
    "left_taper_angle_vertical",
    "right_taper_angle_vertical",
    "avg_taper_angle",
    "taper_angle_diff",
    "left_fit_r2",
    "right_fit_r2",
    "left_fit_error",
    "right_fit_error",
    "left_valid_point_count",
    "right_valid_point_count",
    "left_taper_status",
    "right_taper_status",
    "horizontal_cd_mean_px",
    "horizontal_cd_max_px",
    "horizontal_cd_min_px",
    "horizontal_cd_median_px",
    "horizontal_cd_std_px",
    "horizontal_cd_selected_px",
    "horizontal_cd_selected_method",
    "horizontal_cd_mean",
    "horizontal_cd_max",
    "horizontal_cd_min",
    "horizontal_cd_median",
    "horizontal_cd_std",
    "horizontal_cd_selected",
    "horizontal_cd_valid_count",
    "horizontal_cd_confidence",
    "horizontal_cd_status",
    "vertical_thk_mean_px",
    "vertical_thk_max_px",
    "vertical_thk_min_px",
    "vertical_thk_median_px",
    "vertical_thk_std_px",
    "vertical_thk_selected_px",
    "vertical_thk_selected_method",
    "vertical_thk_mean",
    "vertical_thk_max",
    "vertical_thk_min",
    "vertical_thk_median",
    "vertical_thk_std",
    "vertical_thk_selected",
    "vertical_thk_valid_count",
    "vertical_thk_confidence",
    "vertical_thk_status",
    "overall_confidence",
    "status",
    "warning_message",
    "measurement_source",
    "raw_edge_count",
    "candidate_coverage",
    "selected_point_count",
    "pair_candidate_count",
    "horizontal_cd_raw_edge_count",
    "horizontal_cd_pair_candidate_count",
    "horizontal_cd_scanline_coverage",
    "horizontal_cd_selected_point_count",
    "horizontal_cd_scanned_line_count",
    "vertical_thk_raw_edge_count",
    "vertical_thk_pair_candidate_count",
    "vertical_thk_scanline_coverage",
    "vertical_thk_selected_point_count",
    "vertical_thk_scanned_line_count",
    "left_taper_raw_edge_count",
    "left_taper_scanline_coverage",
    "left_taper_selected_point_count",
    "left_taper_fit_point_count",
    "right_taper_raw_edge_count",
    "right_taper_scanline_coverage",
    "right_taper_selected_point_count",
    "right_taper_fit_point_count",
    "hole_target",
    "hole_cd_horizontal_px",
    "hole_cd_vertical_px",
    "hole_cd_min_feret_px",
    "hole_cd_max_feret_px",
    "hole_cd_equivalent_diameter_px",
    "hole_cd_area_px",
    "hole_cd_perimeter_px",
    "hole_cd_horizontal",
    "hole_cd_vertical",
    "hole_cd_min_feret",
    "hole_cd_max_feret",
    "hole_cd_equivalent_diameter",
    "hole_cd_coverage",
    "hole_cd_mean_radius",
    "hole_cd_radius_std",
    "hole_cd_mean_strength",
    "hole_cd_smoothness",
    "hole_cd_confidence",
    "hole_cd_status",
    "hole_cd_warning_message",
    "hole_cd_ellipse_major_px",
    "hole_cd_ellipse_minor_px",
    "hole_cd_ellipse_angle_deg",
    "hole_cd_ellipse_fit_error",
    "hole_cd_ellipse_aspect_ratio",
    "crater_cd_px",
    "crater_cd",
    "crater_thk_px",
    "crater_thk",
    "crater_thk_mean_px",
    "crater_thk_max_px",
    "crater_thk_min_px",
    "crater_thk_median_px",
    "crater_left_foot_x",
    "crater_left_foot_y",
    "crater_right_foot_x",
    "crater_right_foot_y",
    "crater_baseline_y_left",
    "crater_baseline_y_right",
    "crater_baseline_slope",
    "crater_baseline_intercept",
    "crater_baseline_confidence",
    "crater_center_x",
    "crater_top_y_at_center",
    "crater_baseline_y_at_center",
    "crater_left_taper_angle_horizontal",
    "crater_right_taper_angle_horizontal",
    "crater_left_taper_angle_vertical",
    "crater_right_taper_angle_vertical",
    "crater_avg_taper_angle",
    "crater_taper_angle_diff",
    "crater_left_taper_fit_error",
    "crater_right_taper_fit_error",
    "crater_left_taper_valid_count",
    "crater_right_taper_valid_count",
    "crater_taper_height_percent",
    "crater_left_taper_measure_y",
    "crater_right_taper_measure_y",
    "crater_top_profile_valid_count",
    "crater_top_profile_coverage",
    "crater_top_profile_smoothness",
    "crater_overall_confidence",
    "crater_status",
    "crater_warning_message",
]


def _fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _distance_values(prefix: str, row: Dict[str, object], result: Optional[DistanceResult], settings: MeasurementSettings) -> None:
    fields = [
        "mean_px",
        "max_px",
        "min_px",
        "median_px",
        "std_px",
        "selected_px",
        "selected_method",
        "valid_count",
        "confidence",
        "status",
    ]
    if result is None:
        return
    for field in fields:
        row[f"{prefix}_{field}"] = getattr(result, field)
    scaled = result.scaled(settings.calibration.px_to_real)
    row[f"{prefix}_mean"] = scaled["mean"]
    row[f"{prefix}_max"] = scaled["max"]
    row[f"{prefix}_min"] = scaled["min"]
    row[f"{prefix}_median"] = scaled["median"]
    row[f"{prefix}_std"] = scaled["std"]
    row[f"{prefix}_selected"] = scaled["selected"]
    row[f"{prefix}_raw_edge_count"] = result.raw_edge_count
    row[f"{prefix}_pair_candidate_count"] = result.pair_candidate_count
    row[f"{prefix}_scanline_coverage"] = result.scanline_coverage
    row[f"{prefix}_selected_point_count"] = result.selected_point_count
    row[f"{prefix}_scanned_line_count"] = result.scanned_line_count


def _taper_values(side: str, row: Dict[str, object], result: Optional[TaperSideResult]) -> None:
    if result is None:
        return
    row[f"{side}_taper_angle_horizontal"] = result.angle_horizontal
    row[f"{side}_taper_angle_vertical"] = result.angle_vertical
    row[f"{side}_fit_r2"] = result.fit_r2
    row[f"{side}_fit_error"] = result.fit_error
    row[f"{side}_valid_point_count"] = result.inlier_count or result.valid_point_count
    row[f"{side}_taper_status"] = result.status
    row[f"{side}_taper_raw_edge_count"] = result.raw_edge_count
    row[f"{side}_taper_scanline_coverage"] = result.scanline_coverage
    row[f"{side}_taper_selected_point_count"] = result.selected_point_count
    row[f"{side}_taper_fit_point_count"] = result.fit_point_count


def _hole_cd_values(row: Dict[str, object], result, settings: MeasurementSettings) -> None:
    if result is None:
        return
    scaled = result.scaled(settings.calibration.px_to_real)
    row["hole_target"] = result.target
    row["hole_cd_horizontal_px"] = result.horizontal_px
    row["hole_cd_vertical_px"] = result.vertical_px
    row["hole_cd_min_feret_px"] = result.min_feret_px
    row["hole_cd_max_feret_px"] = result.max_feret_px
    row["hole_cd_equivalent_diameter_px"] = result.equivalent_diameter_px
    row["hole_cd_area_px"] = result.area_px
    row["hole_cd_perimeter_px"] = result.perimeter_px
    row["hole_cd_horizontal"] = scaled["horizontal"]
    row["hole_cd_vertical"] = scaled["vertical"]
    row["hole_cd_min_feret"] = scaled["min_feret"]
    row["hole_cd_max_feret"] = scaled["max_feret"]
    row["hole_cd_equivalent_diameter"] = scaled["equivalent_diameter"]
    row["hole_cd_coverage"] = result.coverage
    row["hole_cd_mean_radius"] = result.mean_radius
    row["hole_cd_radius_std"] = result.radius_std
    row["hole_cd_mean_strength"] = result.mean_strength
    row["hole_cd_smoothness"] = result.smoothness
    row["hole_cd_confidence"] = result.confidence
    row["hole_cd_status"] = result.status
    row["hole_cd_warning_message"] = result.warning_message
    row["hole_cd_ellipse_major_px"] = result.ellipse_major_px
    row["hole_cd_ellipse_minor_px"] = result.ellipse_minor_px
    row["hole_cd_ellipse_angle_deg"] = result.ellipse_angle_deg
    row["hole_cd_ellipse_fit_error"] = result.ellipse_fit_error
    row["hole_cd_ellipse_aspect_ratio"] = result.ellipse_aspect_ratio


def _crater_values(row: Dict[str, object], result) -> None:
    if result is None:
        return
    row["crater_cd_px"] = result.cd_px
    row["crater_cd"] = result.cd
    row["crater_thk_px"] = result.thk_px
    row["crater_thk"] = result.thk
    row["crater_thk_mean_px"] = result.thk_mean_px
    row["crater_thk_max_px"] = result.thk_max_px
    row["crater_thk_min_px"] = result.thk_min_px
    row["crater_thk_median_px"] = result.thk_median_px
    row["crater_left_foot_x"] = result.left_foot_x
    row["crater_left_foot_y"] = result.left_foot_y
    row["crater_right_foot_x"] = result.right_foot_x
    row["crater_right_foot_y"] = result.right_foot_y
    row["crater_baseline_y_left"] = result.baseline_y_left
    row["crater_baseline_y_right"] = result.baseline_y_right
    row["crater_baseline_slope"] = result.baseline_slope
    row["crater_baseline_intercept"] = result.baseline_intercept
    row["crater_baseline_confidence"] = result.baseline_confidence
    row["crater_center_x"] = result.center_x
    row["crater_top_y_at_center"] = result.top_y_at_center
    row["crater_baseline_y_at_center"] = result.baseline_y_at_center
    row["crater_left_taper_angle_horizontal"] = result.left_taper_angle_horizontal
    row["crater_right_taper_angle_horizontal"] = result.right_taper_angle_horizontal
    row["crater_left_taper_angle_vertical"] = result.left_taper_angle_vertical
    row["crater_right_taper_angle_vertical"] = result.right_taper_angle_vertical
    row["crater_avg_taper_angle"] = result.avg_taper_angle
    row["crater_taper_angle_diff"] = result.taper_angle_diff
    row["crater_left_taper_fit_error"] = result.left_taper_fit_error
    row["crater_right_taper_fit_error"] = result.right_taper_fit_error
    row["crater_left_taper_valid_count"] = result.left_taper_valid_count
    row["crater_right_taper_valid_count"] = result.right_taper_valid_count
    row["crater_taper_height_percent"] = result.taper_height_percent
    row["crater_left_taper_measure_y"] = result.left_taper_measure_y
    row["crater_right_taper_measure_y"] = result.right_taper_measure_y
    row["crater_top_profile_valid_count"] = result.top_profile_valid_count
    row["crater_top_profile_coverage"] = result.top_profile_coverage
    row["crater_top_profile_smoothness"] = result.top_profile_smoothness
    row["crater_overall_confidence"] = result.overall_confidence
    row["crater_status"] = result.status
    row["crater_warning_message"] = result.warning_message


def make_result_row(item: ImageItem, settings: MeasurementSettings) -> Dict[str, object]:
    result: Optional[MeasurementResult] = item.result
    roi = settings.roi or ("", "", "", "")
    row: Dict[str, object] = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "file_name": item.file_name,
            "full_path": item.image_path,
            "measurement_type": settings.measurement_type,
            "roi_source_image": settings.roi_source_image,
            "settings_source": settings.settings_source,
            "roi_x1": roi[0],
            "roi_y1": roi[1],
            "roi_x2": roi[2],
            "roi_y2": roi[3],
            "distance_method": settings.distance_method,
            "minimum_grayscale_delta": settings.minimum_grayscale_delta,
            "px_to_real": settings.calibration.px_to_real,
            "unit": settings.calibration.unit,
            "calibration_mode": settings.calibration.mode,
            "calibration_status": settings.calibration.status,
            "detected_scale_bar_px": settings.calibration.detected_scale_bar_px,
            "actual_scale_bar_length": settings.calibration.actual_scale_bar_length,
            "taper_side": settings.taper_side,
        }
    )
    if result is None:
        row["status"] = "Not measured"
        return row

    row["avg_taper_angle"] = result.avg_taper_angle
    row["taper_angle_diff"] = result.taper_angle_diff
    _taper_values("left", row, result.left_taper)
    _taper_values("right", row, result.right_taper)
    _distance_values("horizontal_cd", row, result.horizontal_cd, settings)
    _distance_values("vertical_thk", row, result.vertical_thk, settings)
    _hole_cd_values(row, result.hole_cd, settings)
    _crater_values(row, result.crater)
    row["overall_confidence"] = result.overall_confidence
    row["status"] = result.status
    row["warning_message"] = result.warning_message
    row["measurement_source"] = result.measurement_source
    row["raw_edge_count"] = result.raw_edge_count()
    row["candidate_coverage"] = result.overall_confidence
    row["selected_point_count"] = result.selected_point_count()
    row["pair_candidate_count"] = sum(
        distance.pair_candidate_count
        for distance in (result.horizontal_cd, result.vertical_thk)
        if distance is not None
    )
    return row


def export_results_to_csv(
    path: str,
    image_items: Iterable[ImageItem],
    global_settings: MeasurementSettings,
) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in image_items:
            settings = resolve_effective_settings(item, global_settings)
            row = make_result_row(item, settings)
            writer.writerow({key: _fmt(row.get(key, "")) for key in CSV_COLUMNS})

