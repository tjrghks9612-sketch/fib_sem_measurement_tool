from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import (
    DISTANCE_METHODS,
    DISTANCE_METHOD_BY_LABEL,
    EDGE_SCAN_MODES,
    EDGE_SCAN_MODE_BY_LABEL,
    MEASUREMENT_TYPES,
    MEASUREMENT_TYPE_BY_LABEL,
    MeasurementSettings,
)


class OptionPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_option_changed: Callable[[], None],
        on_apply_calibration: Callable[[], None],
        on_detect_scale_bar: Callable[[], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_option_changed = on_option_changed
        self.on_apply_calibration = on_apply_calibration
        self.on_detect_scale_bar = on_detect_scale_bar
        self._loading = False

        self.measurement_type_var = tk.StringVar(value=MEASUREMENT_TYPES["distance_both"])
        self.taper_side_var = tk.StringVar(value="left")
        self.distance_method_var = tk.StringVar(value=DISTANCE_METHODS["mean"])
        self.edge_scan_mode_var = tk.StringVar(value=EDGE_SCAN_MODES["auto"])
        self.delta_var = tk.StringVar(value="30")
        self.taper_height_var = tk.StringVar(value="50%")
        self.normalize_signal_var = tk.BooleanVar(value=False)
        self.denoise_signal_var = tk.BooleanVar(value=False)
        self.detected_px_var = tk.StringVar(value="")
        self.actual_length_var = tk.StringVar(value="")
        self.unit_var = tk.StringVar(value="um")
        self.show_raw_var = tk.BooleanVar(value=False)
        self.show_selected_var = tk.BooleanVar(value=True)
        self.show_fit_var = tk.BooleanVar(value=True)
        self.show_roi_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)

        self.roi_var = tk.StringVar(value="-")
        self.raw_edge_count_var = tk.StringVar(value="-")
        self.coverage_var = tk.StringVar(value="-")
        self.selected_points_var = tk.StringVar(value="-")
        self.pair_count_var = tk.StringVar(value="-")
        self.threshold_var = tk.StringVar(value="-")
        self.selected_result_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="-")

        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(header, text="검사 설정", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, textvariable=self.status_var, text_color="#59d8ff").pack(side="right")

        self.body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._section_label("측정 설정")
        self._combo_row("측정 모드", self.measurement_type_var, list(MEASUREMENT_TYPES.values()), lambda _v: self._changed())
        self._radio_row("테이퍼 측", self.taper_side_var, [("좌측", "left"), ("우측", "right")])
        self._combo_row("대표값", self.distance_method_var, list(DISTANCE_METHODS.values()), lambda _v: self._changed())
        self._combo_row("탐색 시작", self.edge_scan_mode_var, list(EDGE_SCAN_MODES.values()), lambda _v: self._changed())
        self._switch_row("신호 정규화", self.normalize_signal_var)
        self._switch_row("신호 스무딩", self.denoise_signal_var)
        self._value_row("ROI", self.roi_var)
        row = self._row("최소 변화량")
        self.delta_slider = ctk.CTkSlider(row, from_=1, to=255, command=self._delta_changed)
        self.delta_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(row, textvariable=self.delta_var, width=46, anchor="e", text_color="#dbe7f2").grid(row=0, column=2, sticky="e")
        row = self._row("테이퍼 높이")
        self.taper_height_slider = ctk.CTkSlider(row, from_=0, to=100, command=self._taper_height_changed)
        self.taper_height_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(row, textvariable=self.taper_height_var, width=46, anchor="e", text_color="#dbe7f2").grid(
            row=0, column=2, sticky="e"
        )

        self._section_label("오버레이")
        self._switch_row("원시 후보", self.show_raw_var)
        self._switch_row("선택 경계", self.show_selected_var)
        self._switch_row("피팅 선", self.show_fit_var)
        self._switch_row("ROI", self.show_roi_var)
        self._switch_row("라벨", self.show_labels_var)

        self._section_label("후보 요약")
        self._metric_row("원시 경계 수", self.raw_edge_count_var)
        self._metric_row("신뢰도", self.coverage_var)
        self._metric_row("선택 포인트", self.selected_points_var)
        self._metric_row("쌍 후보", self.pair_count_var)
        self._metric_row("임계값", self.threshold_var)

        self._section_label("선택 결과")
        ctk.CTkLabel(
            self.body,
            textvariable=self.selected_result_var,
            anchor="w",
            justify="left",
            text_color="#f4f8fb",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(fill="x", pady=(0, 8))

        self._section_label("캘리브레이션")
        ctk.CTkButton(self.body, text="스케일바 검출", command=self.on_detect_scale_bar, height=32).pack(fill="x", pady=(4, 6))
        self._entry_row("검출 px", self.detected_px_var)
        self._entry_row("실제 길이", self.actual_length_var)
        self._combo_row("단위", self.unit_var, ["nm", "um", "mm"], lambda _v: self._changed())
        ctk.CTkButton(self.body, text="캘리브레이션 적용", command=self.on_apply_calibration, height=32).pack(fill="x", pady=(6, 10))

    def _section_label(self, text: str) -> None:
        ctk.CTkLabel(self.body, text=text, font=ctk.CTkFont(size=13, weight="bold"), anchor="w", text_color="#f2f6fa").pack(
            fill="x", pady=(12, 5)
        )

    def _row(self, label: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(self.body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, minsize=116)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, minsize=46)
        ctk.CTkLabel(row, text=label, anchor="w", text_color="#aebccb").grid(row=0, column=0, sticky="w", padx=(0, 8))
        return row

    def _combo_row(self, label: str, variable: tk.StringVar, values, command) -> ctk.CTkOptionMenu:
        row = self._row(label)
        menu = ctk.CTkOptionMenu(row, variable=variable, values=values, command=command, height=32)
        menu.grid(row=0, column=1, columnspan=2, sticky="ew")
        return menu

    def _entry_row(self, label: str, variable: tk.StringVar) -> ctk.CTkEntry:
        row = self._row(label)
        entry = ctk.CTkEntry(row, textvariable=variable, height=32)
        entry.grid(row=0, column=1, columnspan=2, sticky="ew")
        entry.bind("<Return>", lambda _event: self._changed())
        entry.bind("<FocusOut>", lambda _event: self._changed())
        return entry

    def _value_row(self, label: str, variable: tk.StringVar) -> None:
        row = self._row(label)
        ctk.CTkLabel(row, textvariable=variable, anchor="e", text_color="#dbe7f2").grid(row=0, column=1, columnspan=2, sticky="e")

    def _metric_row(self, label: str, variable: tk.StringVar) -> None:
        row = ctk.CTkFrame(self.body, fg_color="#0a121b", corner_radius=5)
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, text_color="#8fa1b3", anchor="w").pack(side="left", padx=10, pady=7)
        ctk.CTkLabel(row, textvariable=variable, text_color="#f3f7fb", anchor="e", font=ctk.CTkFont(size=13, weight="bold")).pack(
            side="right", padx=10, pady=7
        )

    def _radio_row(self, label: str, variable: tk.StringVar, values) -> None:
        row = self._row(label)
        box = ctk.CTkFrame(row, fg_color="transparent")
        box.grid(row=0, column=1, columnspan=2, sticky="ew")
        for text, value in values:
            ctk.CTkRadioButton(box, text=text, variable=variable, value=value, command=self._changed).pack(
                side="left", padx=(0, 12)
            )

    def _switch_row(self, label: str, variable: tk.BooleanVar) -> None:
        switch = ctk.CTkSwitch(self.body, text=label, variable=variable, command=self._changed)
        switch.pack(fill="x", pady=3)

    def _delta_changed(self, raw_value: float) -> None:
        value = int(round(float(raw_value)))
        self.delta_var.set(str(value))
        if not self._loading:
            self._changed()

    def _taper_height_changed(self, raw_value: float) -> None:
        value = int(round(float(raw_value)))
        self.taper_height_var.set(f"{value}%")
        if not self._loading:
            self._changed()

    def _changed(self) -> None:
        if not self._loading:
            self.on_option_changed()

    def _float(self, text: str, default: float) -> float:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    def get_settings(self, base: Optional[MeasurementSettings] = None) -> MeasurementSettings:
        settings = base.clone() if base is not None else MeasurementSettings()
        settings.measurement_type = MEASUREMENT_TYPE_BY_LABEL.get(self.measurement_type_var.get(), "distance_both")
        settings.taper_side = self.taper_side_var.get()
        settings.distance_method = DISTANCE_METHOD_BY_LABEL.get(self.distance_method_var.get(), "mean")
        settings.edge_scan_mode = EDGE_SCAN_MODE_BY_LABEL.get(self.edge_scan_mode_var.get(), "auto")
        settings.normalize_grayscale_profiles = bool(self.normalize_signal_var.get())
        settings.denoise_grayscale_profiles = bool(self.denoise_signal_var.get())
        settings.minimum_grayscale_delta = self._float(self.delta_var.get(), settings.minimum_grayscale_delta)
        settings.base_height_pct = self._float(self.taper_height_var.get().replace("%", ""), settings.base_height_pct)
        settings.calibration.unit = self.unit_var.get()
        settings.show_raw_candidates = bool(self.show_raw_var.get())
        settings.show_selected_edges = bool(self.show_selected_var.get())
        settings.show_fit_line = bool(self.show_fit_var.get())
        settings.show_roi = bool(self.show_roi_var.get())
        settings.show_labels = bool(self.show_labels_var.get())
        return settings

    def set_settings(self, settings: MeasurementSettings) -> None:
        self._loading = True
        self.measurement_type_var.set(MEASUREMENT_TYPES.get(settings.measurement_type, MEASUREMENT_TYPES["distance_both"]))
        self.taper_side_var.set(settings.taper_side)
        self.distance_method_var.set(DISTANCE_METHODS.get(settings.distance_method, DISTANCE_METHODS["mean"]))
        self.edge_scan_mode_var.set(EDGE_SCAN_MODES.get(getattr(settings, "edge_scan_mode", "auto"), EDGE_SCAN_MODES["auto"]))
        self.normalize_signal_var.set(bool(getattr(settings, "normalize_grayscale_profiles", False)))
        self.denoise_signal_var.set(bool(getattr(settings, "denoise_grayscale_profiles", False)))
        self.unit_var.set(settings.calibration.unit if settings.calibration.unit != "px" else "um")
        self.detected_px_var.set(
            "" if settings.calibration.detected_scale_bar_px is None else f"{settings.calibration.detected_scale_bar_px:.3f}"
        )
        self.actual_length_var.set(
            ""
            if settings.calibration.actual_scale_bar_length is None
            else f"{settings.calibration.actual_scale_bar_length:.6g}"
        )
        delta = max(1, min(255, int(round(float(settings.minimum_grayscale_delta)))))
        self.delta_var.set(str(delta))
        self.delta_slider.set(delta)
        taper_height = max(0, min(100, int(round(float(getattr(settings, "base_height_pct", 50.0))))))
        self.taper_height_var.set(f"{taper_height}%")
        self.taper_height_slider.set(taper_height)
        self.show_raw_var.set(bool(settings.show_raw_candidates))
        self.show_selected_var.set(bool(settings.show_selected_edges))
        self.show_fit_var.set(bool(settings.show_fit_line))
        self.show_roi_var.set(bool(settings.show_roi))
        self.show_labels_var.set(bool(settings.show_labels))
        self.roi_var.set("-" if settings.roi is None else f"{settings.roi[0]}, {settings.roi[1]}, {settings.roi[2]}, {settings.roi[3]}")
        self.threshold_var.set(f"{settings.minimum_grayscale_delta:.0f} 그레이")
        self._loading = False

    def set_candidate_summary(self, result: Optional[MeasurementResult], settings: MeasurementSettings) -> None:
        if result is None:
            self.raw_edge_count_var.set("-")
            self.coverage_var.set("-")
            self.selected_points_var.set("-")
            self.pair_count_var.set("-")
            self.selected_result_var.set("측정 전")
            self.status_var.set(self._settings_source_label(settings.settings_source))
            return

        raw_edges = result.raw_edge_count()
        selected_points = result.selected_point_count()
        pair_count = sum(
            distance.pair_candidate_count
            for distance in (result.horizontal_cd, result.vertical_thk)
            if distance is not None
        )
        self.raw_edge_count_var.set(str(raw_edges))
        self.coverage_var.set(f"{result.overall_confidence:.0f}%")
        self.selected_points_var.set(str(selected_points))
        self.pair_count_var.set(str(pair_count))
        self.status_var.set(self._status_label(result.status))
        self.selected_result_var.set(self._selected_result_text(result, settings))

    def _settings_source_label(self, source: str) -> str:
        return {
            "global_default": "기본 설정",
            "image_specific": "이미지별 설정",
        }.get(source, source)

    def _status_label(self, status: str) -> str:
        return {
            "OK": "정상",
            "Check": "확인",
            "Review Needed": "검토 필요",
            "Fail": "실패",
        }.get(status, status)

    def _selected_result_text(self, result: MeasurementResult, settings: MeasurementSettings) -> str:
        unit = settings.calibration.unit
        scale = settings.calibration.px_to_real
        if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
            return f"CD {result.horizontal_cd.selected_px * scale:.4g} {unit}"
        if result.vertical_thk and result.vertical_thk.selected_px is not None:
            return f"THK {result.vertical_thk.selected_px * scale:.4g} {unit}"
        if result.avg_taper_angle is not None:
            return f"평균 테이퍼 {result.avg_taper_angle:.2f} deg"
        if result.left_taper and result.left_taper.angle_horizontal is not None:
            return f"좌측 테이퍼 {result.left_taper.angle_horizontal:.2f} deg"
        if result.right_taper and result.right_taper.angle_horizontal is not None:
            return f"우측 테이퍼 {result.right_taper.angle_horizontal:.2f} deg"
        return "선택 결과 없음"

    def get_calibration_inputs(self):
        try:
            pixel_length = float(self.detected_px_var.get())
        except (TypeError, ValueError):
            pixel_length = 0.0
        try:
            actual_length = float(self.actual_length_var.get())
        except (TypeError, ValueError):
            actual_length = 0.0
        return "auto", pixel_length, actual_length, self.unit_var.get()

    def set_detected_scale_bar(self, pixel_length: Optional[float]) -> None:
        self.detected_px_var.set("" if pixel_length is None else f"{pixel_length:.3f}")
