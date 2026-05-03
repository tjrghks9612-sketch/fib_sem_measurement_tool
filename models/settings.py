from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


MEASUREMENT_TYPES = {
    "taper_single": "한쪽 Taper",
    "taper_double": "양쪽 Taper",
    "distance_horizontal": "가로 CD",
    "distance_vertical": "세로 THK",
    "distance_both": "가로+세로 동시",
}

MEASUREMENT_TYPE_BY_LABEL = {label: key for key, label in MEASUREMENT_TYPES.items()}

EDGE_REFERENCES = {
    "outer": "외곽 기준",
    "inner": "내측 기준",
    "center": "경계 중심",
    "strongest": "최대 변화점",
}

EDGE_REFERENCE_BY_LABEL = {label: key for key, label in EDGE_REFERENCES.items()}

DISTANCE_METHODS = {
    "mean": "평균",
    "max": "최대",
    "min": "최소",
}

DISTANCE_METHOD_BY_LABEL = {label: key for key, label in DISTANCE_METHODS.items()}

NOISE_LEVELS = {
    "low": "약",
    "medium": "보통",
    "high": "강",
}

NOISE_LEVEL_BY_LABEL = {label: key for key, label in NOISE_LEVELS.items()}

SCOPES = {
    "current": "현재 이미지",
    "selected": "선택 이미지",
    "group": "현재 그룹",
    "all": "전체 이미지",
}

SCOPE_BY_LABEL = {label: key for key, label in SCOPES.items()}

ROI_APPLY_MODES = {
    "absolute_copy": "절대 좌표 복사",
    "relative_copy": "상대 위치 복사",
    "auto_adjust": "자동 보정(준비)",
}

ROI_APPLY_MODE_BY_LABEL = {label: key for key, label in ROI_APPLY_MODES.items()}

SETTINGS_SOURCES = (
    "image_specific",
    "group_shared",
    "global_default",
    "copied_from_previous",
)

NOISE_PRESET_TABLE = {
    "low": {
        "blur_kernel": 3,
        "median_filter_size": 0,
        "scan_line_count": 15,
        "minimum_valid_line_count": 6,
        "min_valid_line_ratio": 0.40,
        "outlier_rejection_strength": 2.0,
        "sensitivity": 0.62,
        "peak_prominence": 0.10,
    },
    "medium": {
        "blur_kernel": 5,
        "median_filter_size": 3,
        "scan_line_count": 21,
        "minimum_valid_line_count": 8,
        "min_valid_line_ratio": 0.45,
        "outlier_rejection_strength": 1.5,
        "sensitivity": 0.70,
        "peak_prominence": 0.12,
    },
    "high": {
        "blur_kernel": 7,
        "median_filter_size": 5,
        "scan_line_count": 31,
        "minimum_valid_line_count": 14,
        "min_valid_line_ratio": 0.55,
        "outlier_rejection_strength": 1.2,
        "sensitivity": 0.82,
        "peak_prominence": 0.16,
    },
}


@dataclass
class AdvancedSettings:
    blur_kernel: int = 5
    median_filter_size: int = 3
    background_correction_strength: float = 0.0
    sensitivity: float = 0.70
    peak_prominence: float = 0.12
    scan_line_count: int = 21
    minimum_valid_line_count: int = 8
    min_valid_line_ratio: float = 0.45
    outlier_rejection_strength: float = 1.5
    fit_error_threshold: float = 5.0
    confidence_threshold: float = 80.0
    overlay_save_enabled: bool = False
    profile_graph_enabled: bool = False


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
    taper_side: str = "left"
    distance_method: str = "mean"
    edge_reference: str = "inner"
    noise_level: str = "medium"
    roi: Optional[Tuple[int, int, int, int]] = None
    roi_apply_mode: str = "relative_copy"
    roi_source_image: str = ""
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    advanced: AdvancedSettings = field(default_factory=AdvancedSettings)
    custom_option: bool = False
    settings_source: str = "global_default"

    def clone(self) -> "MeasurementSettings":
        return deepcopy(self)

    def apply_noise_preset(self, force: bool = False) -> None:
        if self.custom_option and not force:
            return
        preset = NOISE_PRESET_TABLE.get(self.noise_level, NOISE_PRESET_TABLE["medium"])
        for key, value in preset.items():
            setattr(self.advanced, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_global_settings() -> MeasurementSettings:
    settings = MeasurementSettings()
    settings.apply_noise_preset(force=True)
    return settings


def merge_settings(base: MeasurementSettings, override: Optional[MeasurementSettings]) -> MeasurementSettings:
    merged = base.clone()
    if override is None:
        return merged
    override_dict = override.to_dict()
    unknown_keys = set(override_dict.keys()) - set(MeasurementSettings.__dataclass_fields__.keys())
    if unknown_keys:
        raise ValueError(f"허용되지 않은 설정 키가 있습니다: {sorted(unknown_keys)}")
    advanced_keys = set(AdvancedSettings.__dataclass_fields__.keys())
    calibration_keys = set(CalibrationSettings.__dataclass_fields__.keys())

    for key, value in override_dict.items():
        if key == "advanced":
            nested_unknown = set(value.keys()) - advanced_keys
            if nested_unknown:
                raise ValueError(f"허용되지 않은 고급 설정 키가 있습니다: {sorted(nested_unknown)}")
            merged.advanced = AdvancedSettings(**value)
        elif key == "calibration":
            nested_unknown = set(value.keys()) - calibration_keys
            if nested_unknown:
                raise ValueError(f"허용되지 않은 보정 설정 키가 있습니다: {sorted(nested_unknown)}")
            merged.calibration = CalibrationSettings(**value)
        else:
            setattr(merged, key, value)
    return merged


def resolve_effective_settings(image_item: Any, group_settings: Dict[str, Any], global_settings: MeasurementSettings) -> MeasurementSettings:
    settings = global_settings.clone()
    settings.settings_source = "global_default"

    if getattr(image_item, "group_id", "") and image_item.group_id in group_settings:
        group = group_settings[image_item.group_id]
        settings = merge_settings(settings, group.shared_settings)
        settings.settings_source = "group_shared"

    if getattr(image_item, "settings", None) is not None:
        source = image_item.settings.settings_source or "image_specific"
        settings = merge_settings(settings, image_item.settings)
        settings.settings_source = source if source in SETTINGS_SOURCES else "image_specific"

    return settings
