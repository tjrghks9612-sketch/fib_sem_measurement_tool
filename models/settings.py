from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


MEASUREMENT_TYPES = {
    "taper_single": "Single Taper",
    "taper_double": "Double Taper",
    "distance_horizontal": "Horizontal CD",
    "distance_vertical": "Vertical THK",
    "distance_both": "Horizontal + Vertical",
}

MEASUREMENT_TYPE_BY_LABEL = {label: key for key, label in MEASUREMENT_TYPES.items()}

DISTANCE_METHODS = {
    "mean": "Mean",
    "max": "Max",
    "min": "Min",
}

DISTANCE_METHOD_BY_LABEL = {label: key for key, label in DISTANCE_METHODS.items()}

EDGE_SCAN_MODES = {
    "auto": "Auto",
    "outside_to_center": "Outside -> Center",
    "center_to_outside": "Center -> Outside",
}

EDGE_SCAN_MODE_BY_LABEL = {label: key for key, label in EDGE_SCAN_MODES.items()}

MEASURE_DIRECTIONS = {
    "horizontal": "Horizontal",
    "vertical": "Vertical",
    "both": "Horizontal + Vertical",
}

SETTINGS_SOURCES = (
    "image_specific",
    "global_default",
)


@dataclass
class CalibrationSettings:
    px_to_real: float = 1.0
    unit: str = "px"
    mode: str = "manual"
    detected_scale_bar_px: Optional[float] = None
    actual_scale_bar_length: Optional[float] = None
    status: str = "not_calibrated"
    manual_pixel_length: Optional[float] = None

    def is_calibrated(self) -> bool:
        return self.status == "calibrated" and self.px_to_real > 0 and self.unit != "px"


@dataclass
class MeasurementSettings:
    measurement_type: str = "distance_both"
    measure_direction: str = "both"
    taper_side: str = "left"
    distance_method: str = "mean"
    edge_scan_mode: str = "auto"
    minimum_grayscale_delta: float = 30.0
    max_jump_px: float = 28.0
    base_height_pct: float = 50.0
    left_offset_pct: float = 0.0
    right_offset_pct: float = 0.0
    taper_residual_limit_px: float = 18.0
    cd_left_edge_direction: str = "left_to_center"
    cd_right_edge_direction: str = "right_to_center"
    taper_left_edge_direction: str = "center_to_left"
    taper_right_edge_direction: str = "center_to_right"
    thk_top_edge_direction: str = "top_to_bottom"
    thk_bottom_edge_direction: str = "bottom_to_top"
    roi: Optional[Tuple[int, int, int, int]] = None
    roi_source_image: str = ""
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    show_raw_candidates: bool = False
    show_selected_edges: bool = True
    show_fit_line: bool = True
    show_roi: bool = True
    show_labels: bool = True
    settings_source: str = "global_default"

    def clone(self) -> "MeasurementSettings":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_global_settings() -> MeasurementSettings:
    return MeasurementSettings()


def merge_settings(base: MeasurementSettings, override: Optional[MeasurementSettings]) -> MeasurementSettings:
    merged = base.clone()
    if override is None:
        return merged

    override_dict = override.to_dict()
    unknown_keys = set(override_dict.keys()) - set(MeasurementSettings.__dataclass_fields__.keys())
    if unknown_keys:
        raise ValueError(f"Unknown measurement setting keys: {sorted(unknown_keys)}")

    calibration_keys = set(CalibrationSettings.__dataclass_fields__.keys())
    for key, value in override_dict.items():
        if key == "calibration":
            nested_unknown = set(value.keys()) - calibration_keys
            if nested_unknown:
                raise ValueError(f"Unknown calibration setting keys: {sorted(nested_unknown)}")
            merged.calibration = CalibrationSettings(**value)
        else:
            setattr(merged, key, value)
    return merged


def resolve_effective_settings(
    image_item: Any,
    global_settings: MeasurementSettings,
) -> MeasurementSettings:
    settings = global_settings.clone()
    settings.settings_source = "global_default"

    if getattr(image_item, "settings", None) is not None:
        source = image_item.settings.settings_source or "image_specific"
        settings = merge_settings(settings, image_item.settings)
        settings.settings_source = source if source in SETTINGS_SOURCES else "image_specific"

    return settings
