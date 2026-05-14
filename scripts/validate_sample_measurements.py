from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.core.overlay import draw_overlay
from fib_sem_measurement_tool.export.csv_exporter import CSV_COLUMNS, make_result_row
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


SAMPLE_DIR = Path(r"C:\Users\admin\Downloads\fib_sem_raw_samples")
OUTPUT_DIR = ROOT / "validation_outputs"
OVERLAY_DIR = OUTPUT_DIR / "overlays"


Roi = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SampleSpec:
    file_name: str
    measurement_type: str
    expected_mode: str
    roi: Callable[[int, int], Roi]
    hole_target: str = "inner"
    minimum_delta: float = 55.0


def _roi_ratio(left: float, top: float, right: float, bottom: float) -> Callable[[int, int], Roi]:
    def _make(width: int, height: int) -> Roi:
        return (
            max(0, min(width - 1, int(round(width * left)))),
            max(0, min(height - 1, int(round(height * top)))),
            max(0, min(width - 1, int(round(width * right)))),
            max(0, min(height - 1, int(round(height * bottom)))),
        )

    return _make


SAMPLES = [
    SampleSpec("가로 CD 1.png", "distance_horizontal", "가로 CD", _roi_ratio(0.12, 0.07, 0.88, 0.93)),
    SampleSpec("가로 CD 2.png", "distance_horizontal", "가로 CD", _roi_ratio(0.10, 0.08, 0.90, 0.92)),
    SampleSpec("세로 THK 1.png", "distance_vertical", "세로 THK", _roi_ratio(0.02, 0.25, 0.98, 0.72)),
    SampleSpec("세로 THK 2.png", "distance_vertical", "세로 THK", _roi_ratio(0.02, 0.22, 0.98, 0.68)),
    SampleSpec("가로+세로 1.png", "distance_both", "가로+세로", _roi_ratio(0.10, 0.15, 0.90, 0.86)),
    SampleSpec("가로+세로 2.png", "distance_both", "가로+세로", _roi_ratio(0.10, 0.15, 0.90, 0.86)),
    SampleSpec("Taper 1.png", "taper_double", "Taper", _roi_ratio(0.05, 0.18, 0.95, 0.86)),
    SampleSpec("Taper 2.png", "taper_double", "Taper", _roi_ratio(0.05, 0.18, 0.95, 0.86)),
    SampleSpec("Hole CD 1.png", "hole_cd", "Hole CD", _roi_ratio(0.03, 0.03, 0.97, 0.97), minimum_delta=30.0),
    SampleSpec("Hole CD 2.png", "hole_cd", "Hole CD", _roi_ratio(0.03, 0.03, 0.97, 0.97), minimum_delta=30.0),
    SampleSpec("Crater 1.png", "crater", "Crater", _roi_ratio(0.00, 0.22, 1.00, 0.75)),
    SampleSpec("Crater 2.png", "crater", "Crater", _roi_ratio(0.00, 0.22, 1.00, 0.75)),
]


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _settings_for(spec: SampleSpec, roi: Roi) -> MeasurementSettings:
    settings = MeasurementSettings(
        roi=roi,
        measurement_type=spec.measurement_type,
        hole_target=spec.hole_target,
        minimum_grayscale_delta=spec.minimum_delta,
        show_raw_candidates=False,
        show_selected_edges=True,
        show_fit_line=True,
        show_roi=True,
        show_labels=True,
    )
    if spec.measurement_type == "distance_both":
        settings.measure_direction = "both"
    return settings


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def measured_values(result: MeasurementResult) -> Dict[str, object]:
    values: Dict[str, object] = {}
    if result.horizontal_cd:
        values["horizontal_cd_px"] = result.horizontal_cd.selected_px
    if result.vertical_thk:
        values["vertical_thk_px"] = result.vertical_thk.selected_px
    if result.left_taper:
        values["left_taper_deg"] = result.left_taper.angle_horizontal
        values["left_taper_points"] = result.left_taper.inlier_count or result.left_taper.valid_point_count
    if result.right_taper:
        values["right_taper_deg"] = result.right_taper.angle_horizontal
        values["right_taper_points"] = result.right_taper.inlier_count or result.right_taper.valid_point_count
    if result.hole_cd:
        values["hole_h_px"] = result.hole_cd.horizontal_px
        values["hole_v_px"] = result.hole_cd.vertical_px
        values["hole_target"] = result.hole_cd.target
        values["hole_coverage"] = result.hole_cd.coverage
    if result.crater:
        values["crater_cd_px"] = result.crater.cd_px
        values["crater_thk_px"] = result.crater.thk_px
        values["crater_left_taper_deg"] = result.crater.left_taper_angle_horizontal
        values["crater_right_taper_deg"] = result.crater.right_taper_angle_horizontal
    return values


def judge_result(spec: SampleSpec, result: MeasurementResult) -> tuple[str, str]:
    if result.status == "Fail":
        return "fail", "measurement status is Fail"
    if spec.measurement_type == "distance_horizontal":
        ok = bool(result.horizontal_cd and result.horizontal_cd.selected_px)
        return ("pass" if ok and result.overall_confidence >= 60.0 else "partial", "" if ok else "horizontal CD is empty")
    if spec.measurement_type == "distance_vertical":
        ok = bool(result.vertical_thk and result.vertical_thk.selected_px)
        return ("pass" if ok and result.overall_confidence >= 60.0 else "partial", "" if ok else "vertical THK is empty")
    if spec.measurement_type == "distance_both":
        has_cd = bool(result.horizontal_cd and result.horizontal_cd.selected_px)
        has_thk = bool(result.vertical_thk and result.vertical_thk.selected_px)
        if has_cd and has_thk:
            return ("pass" if result.overall_confidence >= 60.0 else "partial", "" if result.overall_confidence >= 60.0 else "low combined confidence")
        return "partial" if has_cd or has_thk else "fail", "CD or THK is missing"
    if spec.measurement_type == "taper_double":
        left = result.left_taper
        right = result.right_taper
        ok = bool(left and right and left.angle_horizontal is not None and right.angle_horizontal is not None)
        if ok:
            enough = (left.inlier_count or left.valid_point_count) >= 8 and (right.inlier_count or right.valid_point_count) >= 8
            strong = enough and result.overall_confidence >= 60.0
            return ("pass" if strong else "partial", "" if strong else "sidewall point count or confidence is low")
        return "fail", "left or right taper is empty"
    if spec.measurement_type == "hole_cd":
        hole = result.hole_cd
        ok = bool(hole and hole.horizontal_px and hole.vertical_px and hole.contour_points)
        if ok:
            return ("pass" if result.status != "Review Needed" else "partial", "" if result.status != "Review Needed" else "hole boundary needs review")
        return "fail", "hole contour or diameter is empty"
    if spec.measurement_type == "crater":
        crater = result.crater
        ok = bool(
            crater
            and crater.cd_px
            and crater.thk_px
            and crater.left_taper_angle_horizontal is not None
            and crater.right_taper_angle_horizontal is not None
        )
        if ok:
            return ("pass" if result.status != "Review Needed" else "partial", "" if result.status != "Review Needed" else "crater result needs review")
        return "fail", "crater CD, THK, or taper is empty"
    return "fail", "unsupported validation mode"


def write_report(entries: Iterable[Dict[str, object]]) -> None:
    lines = ["# Sample Measurement Validation", ""]
    for entry in entries:
        values = entry["measured_values"]
        value_text = ", ".join(f"{key}={_fmt(value)}" for key, value in values.items()) or "-"
        lines.extend(
            [
                f"## {entry['file_name']}",
                f"- expected_mode: {entry['expected_mode']}",
                f"- measured_values: {value_text}",
                f"- confidence: {_fmt(entry['confidence'])}",
                f"- status: {entry['status']}",
                f"- warning_message: {entry['warning_message'] or '-'}",
                f"- overlay_path: {entry['overlay_path']}",
                f"- validation: {entry['validation']}",
                f"- failure_reason: {entry['failure_reason'] or '-'}",
                "",
            ]
        )
    OUTPUT_DIR.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    OVERLAY_DIR.mkdir(exist_ok=True)
    rows = []
    report_entries = []
    for spec in SAMPLES:
        image_path = SAMPLE_DIR / spec.file_name
        image = read_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read sample image: {image_path}")
        height, width = image.shape[:2]
        roi = spec.roi(width, height)
        settings = _settings_for(spec, roi)
        result = run_measurement(image, settings)
        overlay = draw_overlay(image, roi, result, settings, show_overlay=True, language="ko")
        overlay_path = OVERLAY_DIR / f"{image_path.stem}_overlay.png"
        write_image(overlay_path, overlay)

        item = ImageItem(
            image_path=str(image_path),
            file_name=spec.file_name,
            image_size=(width, height),
            settings=settings,
            result=result,
        )
        row = make_result_row(item, settings)
        rows.append(row)
        validation, reason = judge_result(spec, result)
        report_entries.append(
            {
                "file_name": spec.file_name,
                "expected_mode": spec.expected_mode,
                "measured_values": measured_values(result),
                "confidence": result.overall_confidence,
                "status": result.status,
                "warning_message": result.warning_message,
                "overlay_path": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
                "validation": validation,
                "failure_reason": reason,
            }
        )

    with OUTPUT_DIR.joinpath("results.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_report(report_entries)
    for entry in report_entries:
        print(f"{entry['validation']:7s} {entry['file_name']} {entry['status']} {entry['confidence']:.1f}% {entry['failure_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
