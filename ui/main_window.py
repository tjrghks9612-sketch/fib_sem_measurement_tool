from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.core.calibration import apply_calibration, detect_scale_bar
from fib_sem_measurement_tool.core.image_io import (
    filter_image_paths,
    list_image_files,
    load_image_unicode,
    read_image_metadata,
)
from fib_sem_measurement_tool.core.measurement_runner import run_measurement
from fib_sem_measurement_tool.core.overlay import draw_overlay
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.export.csv_exporter import export_results_to_csv
from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import (
    MEASUREMENT_TYPES,
    MeasurementSettings,
    default_global_settings,
    resolve_effective_settings,
)
from fib_sem_measurement_tool.ui.image_viewer import ImageViewer
from fib_sem_measurement_tool.ui.option_panel import OptionPanel
from fib_sem_measurement_tool.ui.profile_graph import ProfileGraph
from fib_sem_measurement_tool.ui.result_table import ResultTable
from fib_sem_measurement_tool.ui.thumbnail_panel import ThumbnailPanel


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("FIB-SEM Measurement Tool")
        self.geometry("1680x940")
        self.minsize(1280, 760)

        self.global_settings = default_global_settings()
        self.image_items: List[ImageItem] = []
        self.current_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.overlay_enabled = True
        self.scale_bar_bboxes: Dict[str, tuple] = {}
        self.calibration_lines: Dict[str, tuple] = {}
        self._auto_measure_after_id = None
        self.auto_measure_delay_ms = 250
        self._thumbnail_refresh_after_id = None
        self._profile_image_path = ""
        self._last_option_signature = None
        self.image_cache = OrderedDict()
        self.image_cache_limit = 8
        self.render_cache = OrderedDict()
        self.render_cache_limit = 4
        self.status_var = ctk.StringVar(value="Load images, drag an ROI, then measure.")
        self.current_file_var = ctk.StringVar(value="No file selected")

        self._build()

    def _build(self) -> None:
        self.configure(fg_color="#08111a")
        toolbar = ctk.CTkFrame(self, fg_color="#08111a")
        toolbar.pack(fill="x", padx=14, pady=(10, 6))
        ctk.CTkLabel(toolbar, text="FIB/SEM Measurement Tool", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=(4, 18))
        ctk.CTkLabel(toolbar, textvariable=self.current_file_var, text_color="#9dacbc").pack(side="left", padx=(0, 14))
        ctk.CTkButton(toolbar, text="Load Images", width=140, command=self.load_images_dialog).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="Load Folder", width=140, command=self.load_folder_dialog).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="Previous", width=90, fg_color="#142234", command=self.previous_image).pack(side="left", padx=(18, 4))
        ctk.CTkButton(toolbar, text="Next", width=90, fg_color="#142234", command=self.next_image).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="Measure Current", width=140, command=lambda: self.measure_scope("current")).pack(side="left", padx=(18, 4))
        ctk.CTkButton(toolbar, text="Measure All", width=120, fg_color="#203246", command=lambda: self.measure_scope("all")).pack(side="left", padx=4)
        ctk.CTkButton(toolbar, text="Save CSV", width=110, command=self.export_csv).pack(side="right", padx=4)

        main = ctk.CTkFrame(self, fg_color="#08111a")
        main.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        main.grid_columnconfigure(0, weight=3, uniform="main")
        main.grid_columnconfigure(1, weight=7, uniform="main")
        main.grid_columnconfigure(2, weight=4, uniform="main")
        main.grid_rowconfigure(0, weight=1)

        self.thumbnail_panel = ThumbnailPanel(
            main,
            on_select_image=self.select_image,
            on_selection_changed=self.refresh_thumbnail_panel,
        )
        self.thumbnail_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        center = ctk.CTkFrame(main, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=4)
        center.grid_rowconfigure(0, weight=5)
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.viewer = ImageViewer(
            center,
            on_roi_changed=self.on_roi_changed,
            on_calibration_line=self.on_calibration_line,
            on_overlay_toggled=self.on_overlay_toggled,
            on_hover_profile=self.on_profile_hover,
        )
        self.viewer.grid(row=0, column=0, sticky="nsew")
        self.profile_graph = ProfileGraph(center)
        self.profile_graph.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.option_panel = OptionPanel(
            main,
            on_option_changed=self.on_option_changed,
            on_apply_calibration=self.apply_calibration_to_scope,
            on_detect_scale_bar=self.detect_current_scale_bar,
        )
        self.option_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.option_panel.set_settings(self.global_settings)
        self.option_panel.set_candidate_summary(None, self.global_settings)

        self.result_table = ResultTable(self)
        self.result_table.pack(fill="x", padx=14, pady=(0, 8))

        status = ctk.CTkFrame(self, fg_color="#08111a")
        status.pack(fill="x", padx=18, pady=(0, 8))
        ctk.CTkLabel(status, textvariable=self.status_var, anchor="w", text_color="#9dacbc").pack(side="left", fill="x", expand=True)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.update_idletasks()

    def load_images_dialog(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select FIB-SEM Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        self.add_image_paths(paths)

    def load_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.add_image_paths(list_image_files(folder))

    def add_image_paths(self, paths) -> None:
        image_paths = filter_image_paths(paths)
        if not image_paths:
            return
        existing = {item.image_path for item in self.image_items}
        added = 0
        errors = []
        for path in image_paths:
            if path in existing:
                continue
            try:
                image_size, thumbnail = read_image_metadata(path)
                self.image_items.append(ImageItem.from_path(path, image_size, thumbnail))
                added += 1
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if self.current_index < 0 and self.image_items:
            self.current_index = 0
            self.load_current_image()
        self.refresh_all()
        self.set_status(f"Loaded {added} image(s)." + (f" Failed {len(errors)}." if errors else ""))
        if errors:
            messagebox.showwarning("Image Load Failed", "\n".join(errors[:8]))

    def resolve_settings_for_item(self, item: ImageItem) -> MeasurementSettings:
        return resolve_effective_settings(item, self.global_settings)

    def current_item(self) -> Optional[ImageItem]:
        if 0 <= self.current_index < len(self.image_items):
            return self.image_items[self.current_index]
        return None

    def select_image(self, index: int) -> None:
        if not (0 <= index < len(self.image_items)):
            return
        if index == self.current_index and self.current_image is not None:
            return
        self._cancel_auto_measure()
        self.current_index = index
        self.load_current_image()
        self.schedule_thumbnail_panel_refresh()

    def previous_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(max(0, self.current_index - 1))

    def next_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(min(len(self.image_items) - 1, self.current_index + 1))

    def _cancel_auto_measure(self) -> None:
        if self._auto_measure_after_id is not None:
            try:
                self.after_cancel(self._auto_measure_after_id)
            except ValueError:
                pass
            self._auto_measure_after_id = None

    def load_image_cached(self, path: str):
        cached = self.image_cache.get(path)
        if cached is not None:
            self.image_cache.move_to_end(path)
            return cached
        image = load_image_unicode(path)
        self.image_cache[path] = image
        self.image_cache.move_to_end(path)
        while len(self.image_cache) > self.image_cache_limit:
            self.image_cache.popitem(last=False)
        return image

    def load_current_image(self) -> None:
        item = self.current_item()
        if item is None:
            self.current_image = None
            self.current_image_path = ""
            self.current_file_var.set("No file selected")
            self.viewer.clear()
            self.profile_graph.set_image(None)
            self._profile_image_path = ""
            self._last_option_signature = None
            return
        if self.current_image_path != item.image_path:
            self.current_image = self.load_image_cached(item.image_path)
            self.current_image_path = item.image_path
            self._last_option_signature = None
        self.render_current_image()

    def render_current_image(self) -> None:
        item = self.current_item()
        if item is None or self.current_image is None:
            self.viewer.clear()
            return
        settings = self.resolve_settings_for_item(item)
        rendered = self.get_rendered_preview(item, settings)
        measurement_label = MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type)
        status = item.result.status + f" {item.result.overall_confidence:.0f}%" if item.result else settings.settings_source
        title = f"{item.file_name}"
        meta = f"{measurement_label} | {settings.settings_source}"
        self.current_file_var.set(item.file_name)
        self.viewer.set_content(self.current_image, rendered, title, meta, status)
        if self._profile_image_path != item.image_path:
            self.profile_graph.set_image(self.current_image)
            self._profile_image_path = item.image_path
        self.profile_graph.set_result(item.result)
        self.option_panel.set_settings(settings)
        self.option_panel.set_candidate_summary(item.result, settings)

    def _render_cache_key(self, item: ImageItem, settings: MeasurementSettings):
        calibration_line = self.calibration_lines.get(item.image_path)
        scale_bar_bbox = self.scale_bar_bboxes.get(item.image_path)
        calibration = settings.calibration
        return (
            item.image_path,
            id(item.result),
            settings.roi,
            settings.measurement_type,
            settings.measure_direction,
            settings.distance_method,
            settings.taper_side,
            round(float(settings.max_jump_px), 4),
            round(float(settings.base_height_pct), 4),
            round(float(settings.left_offset_pct), 4),
            round(float(settings.right_offset_pct), 4),
            round(float(settings.taper_residual_limit_px), 4),
            settings.taper_left_edge_direction,
            settings.taper_right_edge_direction,
            round(float(calibration.px_to_real), 8),
            calibration.unit,
            self.overlay_enabled,
            settings.show_raw_candidates,
            settings.show_selected_edges,
            settings.show_fit_line,
            settings.show_roi,
            settings.show_labels,
            calibration_line,
            scale_bar_bbox,
        )

    def get_rendered_preview(self, item: ImageItem, settings: MeasurementSettings):
        key = self._render_cache_key(item, settings)
        cached = self.render_cache.get(key)
        if cached is not None:
            self.render_cache.move_to_end(key)
            return cached
        rendered = draw_overlay(
            self.current_image,
            settings.roi,
            item.result,
            settings,
            show_overlay=self.overlay_enabled,
            calibration_line=self.calibration_lines.get(item.image_path),
            scale_bar_bbox=self.scale_bar_bboxes.get(item.image_path),
        )
        self.render_cache[key] = rendered
        self.render_cache.move_to_end(key)
        while len(self.render_cache) > self.render_cache_limit:
            self.render_cache.popitem(last=False)
        return rendered

    def on_profile_hover(self, x: Optional[int], y: Optional[int]) -> None:
        self.profile_graph.update_cursor(x, y)

    def refresh_thumbnail_panel(self) -> None:
        self.thumbnail_panel.refresh(self.image_items, self.current_index, self.resolve_settings_for_item)

    def refresh_result_table(self) -> None:
        self.result_table.refresh(self.image_items, self.resolve_settings_for_item)

    def schedule_thumbnail_panel_refresh(self) -> None:
        if self._thumbnail_refresh_after_id is not None:
            try:
                self.after_cancel(self._thumbnail_refresh_after_id)
            except ValueError:
                pass
        self._thumbnail_refresh_after_id = self.after(320, self._flush_thumbnail_panel_refresh)

    def _flush_thumbnail_panel_refresh(self) -> None:
        self._thumbnail_refresh_after_id = None
        self.refresh_thumbnail_panel()

    def refresh_all(self) -> None:
        self.render_current_image()
        self.refresh_thumbnail_panel()
        self.refresh_result_table()

    def _ensure_item_settings(self, item: ImageItem, source: str = "image_specific") -> MeasurementSettings:
        settings = self.resolve_settings_for_item(item)
        settings.settings_source = source
        item.settings = settings
        return settings

    def _settings_signature(self, settings: MeasurementSettings):
        cal = settings.calibration
        return (
            settings.roi,
            settings.measurement_type,
            settings.measure_direction,
            settings.taper_side,
            settings.distance_method,
            round(float(settings.minimum_grayscale_delta), 4),
            round(float(settings.max_jump_px), 4),
            round(float(settings.base_height_pct), 4),
            round(float(settings.left_offset_pct), 4),
            round(float(settings.right_offset_pct), 4),
            round(float(settings.taper_residual_limit_px), 4),
            settings.taper_left_edge_direction,
            settings.taper_right_edge_direction,
            round(float(cal.px_to_real), 8),
            cal.unit,
            settings.show_raw_candidates,
            settings.show_selected_edges,
            settings.show_fit_line,
            settings.show_roi,
            settings.show_labels,
        )

    def on_option_changed(self) -> None:
        item = self.current_item()
        if item is None:
            self.global_settings = self.option_panel.get_settings(self.global_settings)
            self.global_settings.settings_source = "global_default"
            return
        base = self.resolve_settings_for_item(item)
        settings = self.option_panel.get_settings(base)
        settings.settings_source = "image_specific"
        signature = self._settings_signature(settings)
        if signature == self._last_option_signature:
            return
        self._last_option_signature = signature
        item.settings = settings
        if settings.roi is not None and self.current_image is not None:
            item.result = None
            self.render_current_image()
            self._schedule_current_auto_measure()
        else:
            self.render_current_image()
            self.schedule_thumbnail_panel_refresh()

    def _schedule_current_auto_measure(self) -> None:
        if self._auto_measure_after_id is not None:
            try:
                self.after_cancel(self._auto_measure_after_id)
            except ValueError:
                pass
        self._auto_measure_after_id = self.after(self.auto_measure_delay_ms, self._auto_measure_current)

    def _auto_measure_current(self) -> None:
        self._auto_measure_after_id = None
        item = self.current_item()
        if item is None or self.current_image is None:
            return
        settings = self.resolve_settings_for_item(item)
        if settings.roi is None:
            self.render_current_image()
            self.refresh_thumbnail_panel()
            return
        item.result = run_measurement(self.current_image, settings)
        self.render_current_image()
        self.schedule_thumbnail_panel_refresh()
        self.refresh_result_table()
        self.set_status(f"Options changed: {item.file_name} / {item.result.status} {item.result.overall_confidence:.0f}%")

    def on_roi_changed(self, roi) -> None:
        item = self.current_item()
        if item is None:
            return
        clean_roi = normalize_roi(roi, item.image_size)
        if clean_roi is None:
            self.set_status("ROI is too small.")
            return
        settings = self._ensure_item_settings(item, "image_specific")
        settings.roi = clean_roi
        settings.roi_source_image = item.file_name
        item.result = None
        self.set_status("ROI applied to current image.")
        self.refresh_all()
        if self.current_image is not None:
            self._schedule_current_auto_measure()

    def on_calibration_line(self, line, length: float) -> None:
        item = self.current_item()
        if item is None:
            return
        self.calibration_lines[item.image_path] = line
        self.set_status(f"Manual calibration line: {length:.2f} px")
        self.render_current_image()

    def on_overlay_toggled(self, enabled: bool) -> None:
        self.overlay_enabled = enabled
        self.render_current_image()

    def detect_current_scale_bar(self) -> None:
        item = self.current_item()
        if item is None or self.current_image is None:
            return
        result = detect_scale_bar(self.current_image)
        if result.get("status") == "detected":
            pixel_length = result.get("pixel_length")
            self.option_panel.set_detected_scale_bar(float(pixel_length))
            self.scale_bar_bboxes[item.image_path] = result.get("bbox")
            self.set_status(f"Scale bar candidate detected: {pixel_length:.2f} px")
        else:
            self.option_panel.set_detected_scale_bar(None)
            self.set_status(str(result.get("message", "Scale bar detection failed")))
            messagebox.showinfo("Scale Bar Detection", str(result.get("message", "Scale bar detection failed")))
        self.render_current_image()

    def _targets_for_scope(self, scope: str) -> List[ImageItem]:
        item = self.current_item()
        if scope == "all":
            return list(self.image_items)
        return [item] if item else []

    def apply_calibration_to_scope(self) -> None:
        mode, pixel_length, actual_length, unit = self.option_panel.get_calibration_inputs()
        calibration = apply_calibration(pixel_length, actual_length, unit, mode=mode)
        if calibration.status != "calibrated":
            messagebox.showwarning("캘리브레이션 실패", "스케일바 검출 또는 실제 길이 입력을 확인하세요.")
            return
        item = self.current_item()
        if item is None:
            self.global_settings.calibration = calibration
            self.set_status(f"기본 캘리브레이션 적용 ({calibration.px_to_real:.6g} {unit}/px)")
        else:
            settings = self._ensure_item_settings(item, "image_specific")
            settings.calibration = calibration
            self.set_status(f"{item.file_name}: 캘리브레이션 적용 ({calibration.px_to_real:.6g} {unit}/px)")
        self.refresh_all()

    def measure_scope(self, force_scope: Optional[str] = None) -> None:
        if not self.image_items:
            return
        scope = force_scope or "current"
        targets = self._targets_for_scope(scope)
        if not targets:
            return
        failures = 0
        for idx, item in enumerate(targets, start=1):
            self.set_status(f"Measuring {idx}/{len(targets)}: {item.file_name}")
            image = self.load_image_cached(item.image_path)
            settings = self.resolve_settings_for_item(item)
            item.result = run_measurement(image, settings)
            if item.result.status == "Fail":
                failures += 1
        self.load_current_image()
        self.refresh_thumbnail_panel()
        self.refresh_result_table()
        self.set_status(f"Measured {len(targets)} image(s)." + (f" / Fail {failures}" if failures else ""))

    def export_csv(self) -> None:
        if not self.image_items:
            return
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_results_to_csv(path, self.image_items, self.global_settings)
        self.set_status(f"CSV saved: {path}")
        messagebox.showinfo("Save CSV", "Result CSV saved.")
