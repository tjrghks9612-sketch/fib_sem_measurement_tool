from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import (
    distance_method_key,
    distance_method_label,
    edge_scan_mode_key,
    edge_scan_mode_label,
    measurement_key,
    measurement_label,
    settings_source_label,
    status_label,
    t,
    taper_side_label,
)


MEASUREMENT_KEYS = (
    "taper_single",
    "taper_double",
    "distance_horizontal",
    "distance_vertical",
    "distance_both",
)
DISTANCE_METHOD_KEYS = ("mean", "max", "min")
EDGE_SCAN_MODE_KEYS = ("auto", "outside_to_center", "center_to_outside")


def option_visibility_for_measurement_type(measurement_type: str, boundary_angle_filter_enabled: bool = False) -> dict[str, bool]:
    is_distance = measurement_type in {"distance_horizontal", "distance_vertical", "distance_both"}
    is_taper = measurement_type in {"taper_single", "taper_double"}
    return {
        "taper_side": measurement_type == "taper_single",
        "representative_value": is_distance,
        "edge_scan_start": is_distance or is_taper,
        "minimum_delta": is_distance or is_taper,
        "normalize_signal": is_distance,
        "denoise_signal": is_distance,
        "boundary_angle_filter": is_distance,
        "max_boundary_angle": is_distance and boundary_angle_filter_enabled,
        "taper_height": is_taper,
        "selected_edges": is_distance or is_taper,
        "fit_line": is_taper,
        "ROI": is_distance or is_taper,
        "labels": is_distance or is_taper,
    }


class OptionPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_option_changed: Callable[[], None],
        language: str = "ko",
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_option_changed = on_option_changed
        self.language = language
        self._loading = False
        self._slider_change_after_id = None
        self._current_settings = MeasurementSettings()
        self._current_result: Optional[MeasurementResult] = None
        self._translatable_widgets = []

        self.measurement_type_var = tk.StringVar(value=measurement_label(self.language, "distance_both"))
        self.taper_side_var = tk.StringVar(value="left")
        self.distance_method_var = tk.StringVar(value=distance_method_label(self.language, "mean"))
        self.edge_scan_mode_var = tk.StringVar(value=edge_scan_mode_label(self.language, "auto"))
        self.delta_var = tk.StringVar(value="55")
        self.max_boundary_angle_var = tk.StringVar(value="18°")
        self.taper_height_var = tk.StringVar(value="30%")
        self.normalize_signal_var = tk.BooleanVar(value=False)
        self.denoise_signal_var = tk.BooleanVar(value=False)
        self.boundary_angle_filter_var = tk.BooleanVar(value=False)
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

    def _measurement_type_key(self) -> str:
        return measurement_key(self.measurement_type_var.get(), "distance_both")

    def _is_distance_mode(self) -> bool:
        return self._measurement_type_key() in {"distance_horizontal", "distance_vertical", "distance_both"}

    def _is_taper_mode(self) -> bool:
        return self._measurement_type_key() in {"taper_single", "taper_double"}

    def _rebuild_body_preserving_state(self) -> None:
        current_result = self._current_result
        current_settings = self.get_settings(self._current_settings)
        self._build_body()
        self.set_settings(current_settings)
        self.set_candidate_summary(current_result, current_settings)

    def _measurement_type_changed(self, _value: str) -> None:
        self._rebuild_body_preserving_state()
        self._changed()

    def _boundary_angle_filter_changed(self) -> None:
        self._rebuild_body_preserving_state()
        self._changed()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        self.header_label = ctk.CTkLabel(header, text=t(self.language, "option_header"), font=ctk.CTkFont(size=16, weight="bold"))
        self.header_label.pack(side="left")
        ctk.CTkLabel(header, textvariable=self.status_var, text_color="#59d8ff").pack(side="right")

        self.body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._build_body()

    def _build_body(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        self._translatable_widgets = []
        self.taper_left_radio = None
        self.taper_right_radio = None
        self.distance_method_menu = None
        self.edge_scan_mode_menu = None
        self.max_boundary_angle_slider = None
        self.taper_height_slider = None
        visibility = option_visibility_for_measurement_type(
            self._measurement_type_key(),
            bool(self.boundary_angle_filter_var.get()),
        )

        self._section_label("section_measurement")
        self.measurement_type_menu = self._combo_row(
            "measurement_mode",
            self.measurement_type_var,
            [measurement_label(self.language, key) for key in MEASUREMENT_KEYS],
            self._measurement_type_changed,
        )
        self._value_row("ROI", self.roi_var)

        if visibility["taper_side"]:
            self._radio_row(
                "taper_side",
                self.taper_side_var,
                [(taper_side_label(self.language, "left"), "left"), (taper_side_label(self.language, "right"), "right")],
            )
        if visibility["representative_value"]:
            self.distance_method_menu = self._combo_row(
                "representative_value",
                self.distance_method_var,
                [distance_method_label(self.language, key) for key in DISTANCE_METHOD_KEYS],
                lambda _v: self._changed(),
            )
        if visibility["edge_scan_start"]:
            self.edge_scan_mode_menu = self._combo_row(
                "edge_scan_start",
                self.edge_scan_mode_var,
                [edge_scan_mode_label(self.language, key) for key in EDGE_SCAN_MODE_KEYS],
                lambda _v: self._changed(),
            )
        if visibility["minimum_delta"]:
            row = self._row("minimum_delta")
            self.delta_slider = ctk.CTkSlider(row, from_=1, to=255, command=self._delta_changed)
            self.delta_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            ctk.CTkLabel(row, textvariable=self.delta_var, width=46, anchor="e", text_color="#dbe7f2").grid(row=0, column=2, sticky="e")
        if visibility["normalize_signal"]:
            self._switch_row("normalize_signal", self.normalize_signal_var)
        if visibility["denoise_signal"]:
            self._switch_row("denoise_signal", self.denoise_signal_var)
        if visibility["boundary_angle_filter"]:
            self._switch_row("boundary_angle_filter", self.boundary_angle_filter_var, command=self._boundary_angle_filter_changed)
        if visibility["max_boundary_angle"]:
            row = self._row("max_boundary_angle")
            self.max_boundary_angle_slider = ctk.CTkSlider(row, from_=1, to=45, command=self._max_boundary_angle_changed)
            self.max_boundary_angle_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            ctk.CTkLabel(row, textvariable=self.max_boundary_angle_var, width=46, anchor="e", text_color="#dbe7f2").grid(
                row=0, column=2, sticky="e"
            )
        if visibility["taper_height"]:
            row = self._row("taper_height")
            self.taper_height_slider = ctk.CTkSlider(row, from_=0, to=100, command=self._taper_height_changed)
            self.taper_height_slider.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            ctk.CTkLabel(row, textvariable=self.taper_height_var, width=46, anchor="e", text_color="#dbe7f2").grid(
                row=0, column=2, sticky="e"
            )

        self._section_label("section_overlay")
        if visibility["selected_edges"]:
            self._switch_row("selected_edges", self.show_selected_var)
        if visibility["fit_line"]:
            self._switch_row("fit_line", self.show_fit_var)
        if visibility["ROI"]:
            self._switch_row("ROI", self.show_roi_var)
        if visibility["labels"]:
            self._switch_row("labels", self.show_labels_var)

        self._section_label("section_candidate_summary")
        self._metric_row("raw_edge_count", self.raw_edge_count_var)
        self._metric_row("confidence", self.coverage_var)
        self._metric_row("selected_points", self.selected_points_var)
        self._metric_row("pair_candidates", self.pair_count_var)
        self._metric_row("threshold", self.threshold_var)

        self._section_label("section_selected_result")
        ctk.CTkLabel(
            self.body,
            textvariable=self.selected_result_var,
            anchor="w",
            justify="left",
            text_color="#f4f8fb",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(fill="x", pady=(0, 8))

    def _section_label(self, key: str) -> None:
        label = ctk.CTkLabel(self.body, text=t(self.language, key), font=ctk.CTkFont(size=13, weight="bold"), anchor="w", text_color="#f2f6fa")
        label.pack(fill="x", pady=(12, 5))
        self._translatable_widgets.append((label, key))

    def _row(self, label_key: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(self.body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        row.grid_columnconfigure(0, minsize=116)
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(2, minsize=46)
        label = ctk.CTkLabel(row, text=t(self.language, label_key), anchor="w", text_color="#aebccb")
        label.grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self._translatable_widgets.append((label, label_key))
        return row

    def _combo_row(self, label_key: str, variable: tk.StringVar, values, command) -> ctk.CTkOptionMenu:
        row = self._row(label_key)
        menu = ctk.CTkOptionMenu(row, variable=variable, values=values, command=command, height=32)
        menu.grid(row=0, column=1, columnspan=2, sticky="ew")
        return menu

    def _entry_row(self, label_key: str, variable: tk.StringVar) -> ctk.CTkEntry:
        row = self._row(label_key)
        entry = ctk.CTkEntry(row, textvariable=variable, height=32)
        entry.grid(row=0, column=1, columnspan=2, sticky="ew")
        entry.bind("<Return>", lambda _event: self._changed())
        entry.bind("<FocusOut>", lambda _event: self._changed())
        return entry

    def _value_row(self, label_key: str, variable: tk.StringVar) -> None:
        row = self._row(label_key)
        ctk.CTkLabel(row, textvariable=variable, anchor="e", text_color="#dbe7f2").grid(row=0, column=1, columnspan=2, sticky="e")

    def _metric_row(self, label_key: str, variable: tk.StringVar) -> None:
        row = ctk.CTkFrame(self.body, fg_color="#0a121b", corner_radius=5)
        row.pack(fill="x", pady=3)
        label = ctk.CTkLabel(row, text=t(self.language, label_key), text_color="#8fa1b3", anchor="w")
        label.pack(side="left", padx=10, pady=7)
        self._translatable_widgets.append((label, label_key))
        ctk.CTkLabel(row, textvariable=variable, text_color="#f3f7fb", anchor="e", font=ctk.CTkFont(size=13, weight="bold")).pack(
            side="right", padx=10, pady=7
        )

    def _radio_row(self, label_key: str, variable: tk.StringVar, values) -> None:
        row = self._row(label_key)
        box = ctk.CTkFrame(row, fg_color="transparent")
        box.grid(row=0, column=1, columnspan=2, sticky="ew")
        for text, value in values:
            button = ctk.CTkRadioButton(box, text=text, variable=variable, value=value, command=self._changed)
            button.pack(
                side="left", padx=(0, 12)
            )
            if value == "left":
                self.taper_left_radio = button
            elif value == "right":
                self.taper_right_radio = button

    def _switch_row(self, label_key: str, variable: tk.BooleanVar, command=None) -> None:
        switch = ctk.CTkSwitch(self.body, text=t(self.language, label_key), variable=variable, command=command or self._changed)
        switch.pack(fill="x", pady=3)
        self._translatable_widgets.append((switch, label_key))

    def _delta_changed(self, raw_value: float) -> None:
        value = int(round(float(raw_value)))
        self.delta_var.set(str(value))
        if not self._loading:
            self._schedule_slider_changed()

    def _taper_height_changed(self, raw_value: float) -> None:
        value = int(round(float(raw_value)))
        self.taper_height_var.set(f"{value}%")
        if not self._loading:
            self._schedule_slider_changed()

    def _max_boundary_angle_changed(self, raw_value: float) -> None:
        value = int(round(float(raw_value)))
        self.max_boundary_angle_var.set(f"{value}°")
        if not self._loading:
            self._schedule_slider_changed()

    def _schedule_slider_changed(self) -> None:
        if self._slider_change_after_id is not None:
            try:
                self.after_cancel(self._slider_change_after_id)
            except ValueError:
                pass
        self._slider_change_after_id = self.after(120, self._flush_slider_changed)

    def _flush_slider_changed(self) -> None:
        self._slider_change_after_id = None
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
        settings.measurement_type = measurement_key(self.measurement_type_var.get(), "distance_both")
        settings.taper_side = self.taper_side_var.get()
        settings.distance_method = distance_method_key(self.distance_method_var.get(), "mean")
        settings.edge_scan_mode = edge_scan_mode_key(self.edge_scan_mode_var.get(), "auto")
        settings.normalize_grayscale_profiles = bool(self.normalize_signal_var.get())
        settings.denoise_grayscale_profiles = bool(self.denoise_signal_var.get())
        settings.minimum_grayscale_delta = self._float(self.delta_var.get(), settings.minimum_grayscale_delta)
        settings.filter_cd_thk_by_boundary_angle = bool(self.boundary_angle_filter_var.get())
        settings.max_cd_thk_boundary_angle_deg = self._float(
            self.max_boundary_angle_var.get().replace("°", ""),
            settings.max_cd_thk_boundary_angle_deg,
        )
        settings.base_height_pct = self._float(self.taper_height_var.get().replace("%", ""), settings.base_height_pct)
        settings.show_raw_candidates = False
        settings.show_selected_edges = bool(self.show_selected_var.get())
        settings.show_fit_line = bool(self.show_fit_var.get())
        settings.show_roi = bool(self.show_roi_var.get())
        settings.show_labels = bool(self.show_labels_var.get())
        return settings

    def set_settings(self, settings: MeasurementSettings) -> None:
        self._loading = True
        self._current_settings = settings.clone()
        if settings.measurement_type not in MEASUREMENT_KEYS:
            settings = settings.clone()
            settings.measurement_type = "distance_both"
        self.measurement_type_var.set(measurement_label(self.language, settings.measurement_type))
        self.taper_side_var.set(settings.taper_side)
        self.distance_method_var.set(distance_method_label(self.language, settings.distance_method))
        self.edge_scan_mode_var.set(edge_scan_mode_label(self.language, getattr(settings, "edge_scan_mode", "auto")))
        self.normalize_signal_var.set(bool(getattr(settings, "normalize_grayscale_profiles", False)))
        self.denoise_signal_var.set(bool(getattr(settings, "denoise_grayscale_profiles", False)))
        delta = max(1, min(255, int(round(float(settings.minimum_grayscale_delta)))))
        self.delta_var.set(str(delta))
        if hasattr(self, "delta_slider"):
            self.delta_slider.set(delta)
        self.boundary_angle_filter_var.set(bool(getattr(settings, "filter_cd_thk_by_boundary_angle", False)))
        max_angle = max(1, min(45, int(round(float(getattr(settings, "max_cd_thk_boundary_angle_deg", 18.0))))))
        self.max_boundary_angle_var.set(f"{max_angle}°")
        if self.max_boundary_angle_slider is not None:
            self.max_boundary_angle_slider.set(max_angle)
        taper_height = max(0, min(100, int(round(float(getattr(settings, "base_height_pct", 30.0))))))
        self.taper_height_var.set(f"{taper_height}%")
        if self.taper_height_slider is not None:
            self.taper_height_slider.set(taper_height)
        self.show_selected_var.set(bool(settings.show_selected_edges))
        self.show_fit_var.set(bool(settings.show_fit_line))
        self.show_roi_var.set(bool(settings.show_roi))
        self.show_labels_var.set(bool(settings.show_labels))
        self.roi_var.set("-" if settings.roi is None else f"{settings.roi[0]}, {settings.roi[1]}, {settings.roi[2]}, {settings.roi[3]}")
        self.threshold_var.set(f"{settings.minimum_grayscale_delta:.0f} {t(self.language, 'gray_suffix')}")
        self._loading = False

    def set_candidate_summary(self, result: Optional[MeasurementResult], settings: MeasurementSettings) -> None:
        self._current_result = result
        self._current_settings = settings.clone()
        if result is None:
            self.raw_edge_count_var.set("-")
            self.coverage_var.set("-")
            self.selected_points_var.set("-")
            self.pair_count_var.set("-")
            self.selected_result_var.set(t(self.language, "before_measurement"))
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
        return settings_source_label(self.language, source)

    def _status_label(self, status: str) -> str:
        return status_label(self.language, status)

    def _selected_result_text(self, result: MeasurementResult, settings: MeasurementSettings) -> str:
        unit = settings.calibration.unit
        scale = settings.calibration.px_to_real
        if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
            return f"CD {result.horizontal_cd.selected_px * scale:.4g} {unit}"
        if result.vertical_thk and result.vertical_thk.selected_px is not None:
            return f"THK {result.vertical_thk.selected_px * scale:.4g} {unit}"
        if result.avg_taper_angle is not None:
            return f"{t(self.language, 'average_taper')} {result.avg_taper_angle:.2f} deg"
        if result.left_taper and result.left_taper.angle_horizontal is not None:
            return f"{t(self.language, 'left_taper')} {result.left_taper.angle_horizontal:.2f} deg"
        if result.right_taper and result.right_taper.angle_horizontal is not None:
            return f"{t(self.language, 'right_taper')} {result.right_taper.angle_horizontal:.2f} deg"
        return t(self.language, "no_selected_result")

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        self._loading = True
        try:
            current_settings = self.get_settings(self._current_settings)
            self.language = language
            self.header_label.configure(text=t(self.language, "option_header"))
            for widget, key in self._translatable_widgets:
                widget.configure(text=t(self.language, key))
            if self.taper_left_radio is not None:
                self.taper_left_radio.configure(text=taper_side_label(self.language, "left"))
            if self.taper_right_radio is not None:
                self.taper_right_radio.configure(text=taper_side_label(self.language, "right"))
            self.measurement_type_menu.configure(values=[measurement_label(self.language, key) for key in MEASUREMENT_KEYS])
            if self.distance_method_menu is not None:
                self.distance_method_menu.configure(values=[distance_method_label(self.language, key) for key in DISTANCE_METHOD_KEYS])
            if self.edge_scan_mode_menu is not None:
                self.edge_scan_mode_menu.configure(values=[edge_scan_mode_label(self.language, key) for key in EDGE_SCAN_MODE_KEYS])
            self.set_settings(current_settings)
            self.set_candidate_summary(self._current_result, current_settings)
        finally:
            self._loading = False
