from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk
from PIL import Image, ImageDraw

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
    MeasurementSettings,
    default_global_settings,
    resolve_effective_settings,
)
from fib_sem_measurement_tool.ui.image_viewer import ImageViewer
from fib_sem_measurement_tool.ui.i18n import (
    LANGUAGES,
    language_code,
    language_label,
    measurement_label,
    settings_source_label,
    status_label,
    t,
)
from fib_sem_measurement_tool.ui.option_panel import OptionPanel
from fib_sem_measurement_tool.ui.profile_graph import ProfileGraph
from fib_sem_measurement_tool.ui.thumbnail_panel import ThumbnailPanel


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.language = "ko"
        self.title(t(self.language, "app_title"))
        self.geometry("1680x940")
        self.minsize(1280, 760)

        self.global_settings = default_global_settings()
        self.image_items: List[ImageItem] = []
        self.current_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.overlay_enabled = True
        self.scale_bar_bboxes: Dict[str, tuple] = {}
        self._profile_image_path = ""
        self._last_option_signature = None
        self.image_cache = OrderedDict()
        self.image_cache_limit = 8
        self.render_cache = OrderedDict()
        self.render_cache_limit = 4
        self.thumbnail_overlay_cache = OrderedDict()
        self.thumbnail_overlay_cache_limit = 32
        self.status_var = ctk.StringVar(value=t(self.language, "initial_status"))
        self.current_file_var = ctk.StringVar(value=t(self.language, "current_file_none"))
        self.language_var = ctk.StringVar(value=language_label(self.language))

        self._build()

    def _build(self) -> None:
        self.configure(fg_color="#08111a")
        toolbar = ctk.CTkFrame(self, fg_color="#08111a")
        toolbar.pack(fill="x", padx=14, pady=(10, 6))
        self.toolbar_title_label = ctk.CTkLabel(toolbar, text=t(self.language, "toolbar_title"), font=ctk.CTkFont(size=20, weight="bold"))
        self.toolbar_title_label.pack(side="left", padx=(4, 18))
        ctk.CTkLabel(toolbar, textvariable=self.current_file_var, text_color="#9dacbc").pack(side="left", padx=(0, 14))
        self.load_images_button = ctk.CTkButton(toolbar, text=t(self.language, "load_images"), width=140, command=self.load_images_dialog)
        self.load_images_button.pack(side="left", padx=4)
        self.load_folder_button = ctk.CTkButton(toolbar, text=t(self.language, "load_folder"), width=140, command=self.load_folder_dialog)
        self.load_folder_button.pack(side="left", padx=4)
        self.reset_images_button = ctk.CTkButton(toolbar, text=t(self.language, "reset_images"), width=120, fg_color="#203246", command=self.reset_images)
        self.reset_images_button.pack(side="left", padx=4)
        self.previous_button = ctk.CTkButton(toolbar, text=t(self.language, "previous"), width=90, fg_color="#142234", command=self.previous_image)
        self.previous_button.pack(side="left", padx=(18, 4))
        self.next_button = ctk.CTkButton(toolbar, text=t(self.language, "next"), width=90, fg_color="#142234", command=self.next_image)
        self.next_button.pack(side="left", padx=4)
        self.measure_current_button = ctk.CTkButton(toolbar, text=t(self.language, "measure_current"), width=140, command=lambda: self.measure_scope("current"))
        self.measure_current_button.pack(side="left", padx=(18, 4))
        self.measure_all_button = ctk.CTkButton(toolbar, text=t(self.language, "measure_all"), width=120, fg_color="#203246", command=lambda: self.measure_scope("all"))
        self.measure_all_button.pack(side="left", padx=4)
        self.csv_button = ctk.CTkButton(toolbar, text=t(self.language, "save_csv"), width=110, command=self.export_csv)
        self.csv_button.pack(side="right", padx=4)
        self.language_menu = ctk.CTkOptionMenu(
            toolbar,
            variable=self.language_var,
            values=list(LANGUAGES.values()),
            width=118,
            command=self.on_language_changed,
        )
        self.language_menu.pack(side="right", padx=(4, 12))
        self.language_label_widget = ctk.CTkLabel(toolbar, text=t(self.language, "language"), text_color="#9dacbc")
        self.language_label_widget.pack(side="right", padx=(4, 4))

        main = ctk.CTkFrame(self, fg_color="#08111a")
        main.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        main.grid_columnconfigure(0, weight=3, uniform="main")
        main.grid_columnconfigure(1, weight=7, uniform="main")
        main.grid_columnconfigure(2, weight=4, uniform="main")
        main.grid_rowconfigure(0, weight=1)

        self.thumbnail_panel = ThumbnailPanel(
            main,
            on_select_image=self.select_image,
            on_selection_changed=lambda: None,
            language=self.language,
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
            on_overlay_toggled=self.on_overlay_toggled,
            on_hover_profile=self.on_profile_hover,
            language=self.language,
        )
        self.viewer.grid(row=0, column=0, sticky="nsew")
        self.profile_graph = ProfileGraph(center, language=self.language)
        self.profile_graph.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.option_panel = OptionPanel(
            main,
            on_option_changed=self.on_option_changed,
            on_apply_calibration=self.apply_calibration_to_scope,
            on_detect_scale_bar=self.detect_current_scale_bar,
            language=self.language,
        )
        self.option_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.option_panel.set_settings(self.global_settings)
        self.option_panel.set_candidate_summary(None, self.global_settings)

        status = ctk.CTkFrame(self, fg_color="#08111a")
        status.pack(fill="x", padx=18, pady=(0, 8))
        ctk.CTkLabel(status, textvariable=self.status_var, anchor="w", text_color="#9dacbc").pack(side="left", fill="x", expand=True)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _status_label(self, status: str) -> str:
        return status_label(self.language, status)

    def _settings_source_label(self, source: str) -> str:
        return settings_source_label(self.language, source)

    def on_language_changed(self, label: str) -> None:
        language = language_code(label)
        if language == self.language:
            return
        self.language = language
        self.title(t(self.language, "app_title"))
        self.language_var.set(language_label(self.language))
        self.toolbar_title_label.configure(text=t(self.language, "toolbar_title"))
        self.load_images_button.configure(text=t(self.language, "load_images"))
        self.load_folder_button.configure(text=t(self.language, "load_folder"))
        self.reset_images_button.configure(text=t(self.language, "reset_images"))
        self.previous_button.configure(text=t(self.language, "previous"))
        self.next_button.configure(text=t(self.language, "next"))
        self.measure_current_button.configure(text=t(self.language, "measure_current"))
        self.measure_all_button.configure(text=t(self.language, "measure_all"))
        self.csv_button.configure(text=t(self.language, "save_csv"))
        self.language_label_widget.configure(text=t(self.language, "language"))
        if self.current_item() is None:
            self.current_file_var.set(t(self.language, "current_file_none"))
            self.status_var.set(t(self.language, "initial_status"))
        self.option_panel.set_language(self.language)
        self.thumbnail_panel.set_language(self.language)
        self.viewer.set_language(self.language)
        self.profile_graph.set_language(self.language)
        self.render_current_image(update_option_settings=False)
        self.refresh_thumbnail_panel()

    def load_images_dialog(self) -> None:
        paths = filedialog.askopenfilenames(
            title=t(self.language, "image_file_dialog_title"),
            filetypes=[(t(self.language, "image_files"), "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), (t(self.language, "all_files"), "*.*")],
        )
        self.add_image_paths(paths)

    def load_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(title=t(self.language, "folder_dialog_title"))
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
        self.set_status(t(self.language, "loaded_images").format(count=added) + (t(self.language, "load_failures").format(count=len(errors)) if errors else ""))
        if errors:
            messagebox.showwarning(t(self.language, "load_image_failed"), "\n".join(errors[:8]))

    def reset_images(self) -> None:
        self.image_items = []
        self.current_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.scale_bar_bboxes.clear()
        self.image_cache.clear()
        self.render_cache.clear()
        self.thumbnail_overlay_cache.clear()
        self._profile_image_path = ""
        self._last_option_signature = None
        self.current_file_var.set(t(self.language, "current_file_none"))
        self.viewer.clear()
        self.profile_graph.set_image(None)
        self.thumbnail_panel.refresh(
            self.image_items,
            self.current_index,
            self.resolve_settings_for_item,
            self.get_thumbnail_preview,
        )
        self.option_panel.set_settings(self.global_settings)
        self.option_panel.set_candidate_summary(None, self.global_settings)
        self.refresh_result_table()
        self.set_status(t(self.language, "initial_status"))

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
        previous_index = self.current_index
        self.current_index = index
        self.load_current_image()
        if previous_index != self.current_index:
            self.thumbnail_panel.set_current_index(self.current_index)

    def previous_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(max(0, self.current_index - 1))

    def next_image(self) -> None:
        if not self.image_items:
            return
        self.select_image(min(len(self.image_items) - 1, self.current_index + 1))

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
            self.current_file_var.set(t(self.language, "current_file_none"))
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

    def render_current_image(self, update_option_settings: bool = True) -> None:
        item = self.current_item()
        if item is None or self.current_image is None:
            self.viewer.clear()
            return
        settings = self.resolve_settings_for_item(item)
        rendered = self.get_rendered_preview(item, settings)
        display_measurement_label = measurement_label(self.language, settings.measurement_type)
        status = (
            self._status_label(item.result.status) + f" {item.result.overall_confidence:.0f}%"
            if item.result
            else self._settings_source_label(settings.settings_source)
        )
        title = f"{item.file_name}"
        meta = f"{display_measurement_label} | {self._settings_source_label(settings.settings_source)}"
        self.current_file_var.set(item.file_name)
        self.viewer.set_content(self.current_image, rendered, title, meta, status)
        self.profile_graph.set_context(self.current_image, settings, item.result)
        self._profile_image_path = item.image_path
        if update_option_settings:
            self.option_panel.set_settings(settings)
        self.option_panel.set_candidate_summary(item.result, settings)

    def _render_cache_key(self, item: ImageItem, settings: MeasurementSettings):
        scale_bar_bbox = self.scale_bar_bboxes.get(item.image_path)
        calibration = settings.calibration
        return (
            item.image_path,
            id(item.result),
            settings.roi,
            settings.measurement_type,
            settings.measure_direction,
            settings.distance_method,
            settings.edge_scan_mode,
            settings.normalize_grayscale_profiles,
            settings.denoise_grayscale_profiles,
            settings.profile_denoise_window,
            round(float(settings.profile_denoise_range_sigma), 4),
            round(float(settings.normalize_low_percentile), 4),
            round(float(settings.normalize_high_percentile), 4),
            round(float(settings.normalize_min_span), 4),
            int(settings.max_profile_candidates_per_scanline),
            bool(getattr(settings, "filter_cd_thk_by_boundary_angle", False)),
            round(float(getattr(settings, "max_cd_thk_boundary_angle_deg", 18.0)), 4),
            int(getattr(settings, "cd_thk_boundary_angle_window", 9)),
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
            settings.show_selected_edges,
            settings.show_fit_line,
            settings.show_roi,
            settings.show_labels,
            self.language,
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
            scale_bar_bbox=self.scale_bar_bboxes.get(item.image_path),
            language=self.language,
        )
        self.render_cache[key] = rendered
        self.render_cache.move_to_end(key)
        while len(self.render_cache) > self.render_cache_limit:
            self.render_cache.popitem(last=False)
        return rendered

    def _thumbnail_overlay_cache_key(self, item: ImageItem, settings: MeasurementSettings):
        return (
            item.image_path,
            id(item.result),
            settings.roi,
            settings.measurement_type,
            settings.show_selected_edges,
            settings.show_fit_line,
            settings.show_roi,
            round(float(settings.base_height_pct), 4),
            round(float(settings.left_offset_pct), 4),
            round(float(settings.right_offset_pct), 4),
            self.overlay_enabled,
            item.thumbnail.size if item.thumbnail is not None else None,
        )

    def get_thumbnail_preview(self, item: ImageItem, settings: MeasurementSettings):
        if settings.roi is None and item.result is None:
            return item.thumbnail
        key = self._thumbnail_overlay_cache_key(item, settings)
        cached = self.thumbnail_overlay_cache.get(key)
        if cached is not None:
            self.thumbnail_overlay_cache.move_to_end(key)
            return cached

        pil = self._draw_fast_thumbnail_overlay(item, settings)
        self.thumbnail_overlay_cache[key] = pil
        self.thumbnail_overlay_cache.move_to_end(key)
        while len(self.thumbnail_overlay_cache) > self.thumbnail_overlay_cache_limit:
            self.thumbnail_overlay_cache.popitem(last=False)
        return pil

    def _draw_fast_thumbnail_overlay(self, item: ImageItem, settings: MeasurementSettings) -> Image.Image:
        source = item.thumbnail
        if source is None:
            return Image.new("RGB", (124, 80), "#101820")
        thumb = source.convert("RGB").copy()
        draw = ImageDraw.Draw(thumb)
        scale_x = thumb.width / max(1, item.image_size[0])
        scale_y = thumb.height / max(1, item.image_size[1])

        def point(x: float, y: float) -> tuple[int, int]:
            return int(round(x * scale_x)), int(round(y * scale_y))

        def taper_sides() -> tuple[str, ...]:
            if settings.measurement_type == "taper_double":
                return ("left", "right")
            if settings.measurement_type == "taper_single":
                return (settings.taper_side if settings.taper_side in ("left", "right") else "left",)
            return ()

        result = item.result

        def taper_target_point(side: str) -> Optional[tuple[float, float]]:
            if result is None:
                return None
            taper = result.right_taper if side == "right" else result.left_taper
            if taper is None or not taper.fit_line:
                return None
            fit_x1, fit_y1, fit_x2, fit_y2 = taper.fit_line
            y_min, y_max = min(float(fit_y1), float(fit_y2)), max(float(fit_y1), float(fit_y2))
            if y_max - y_min <= 1e-6:
                return float((fit_x1 + fit_x2) / 2.0), float(y_min)
            base_pct = max(0.0, min(100.0, float(getattr(settings, "base_height_pct", 50.0))))
            offset_pct = float(
                getattr(settings, "right_offset_pct", 0.0)
                if side == "right"
                else getattr(settings, "left_offset_pct", 0.0)
            )
            target_pct = max(0.0, min(100.0, base_pct + offset_pct))
            target_y = float(y_max) - (float(y_max - y_min) * target_pct / 100.0)
            ratio = (target_y - float(fit_y1)) / (float(fit_y2) - float(fit_y1))
            target_x = float(fit_x1) + (float(fit_x2) - float(fit_x1)) * ratio
            return target_x, target_y

        if settings.roi is not None:
            x1, y1, x2, y2 = settings.roi
            if settings.show_roi:
                draw.rectangle([point(x1, y1), point(x2, y2)], outline=(0, 210, 255), width=1)
            drawn_rows = set()
            for side in taper_sides():
                target = taper_target_point(side)
                if target is None:
                    continue
                target_x, target_y_float = target
                target_y = int(round(target_y_float))
                if target_y in drawn_rows:
                    continue
                drawn_rows.add(target_y)
                guide_half = max(12, min(28, int(abs(x2 - x1) * scale_x * 0.18)))
                cx, cy = point(target_x, target_y)
                draw.line([(cx - guide_half, cy), (cx + guide_half, cy)], fill=(20, 20, 26), width=4)
                draw.line([(cx - guide_half, cy), (cx + guide_half, cy)], fill=(210, 190, 255), width=2)
                draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=(20, 20, 26))
                draw.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], fill=(210, 190, 255), outline=(20, 20, 26))

        if result is not None and settings.show_selected_edges:
            for measurement, color in (
                (result.horizontal_cd, (255, 164, 55)),
                (result.vertical_thk, (80, 245, 120)),
            ):
                if measurement is None:
                    continue
                first_points = [point(pair.first.image_x, pair.first.image_y) for pair in measurement.selected_pairs]
                second_points = [point(pair.second.image_x, pair.second.image_y) for pair in measurement.selected_pairs]
                if len(first_points) >= 2:
                    draw.line(first_points, fill=color, width=2)
                if len(second_points) >= 2:
                    draw.line(second_points, fill=color, width=2)

            for taper, color in (
                (result.left_taper, (255, 220, 75)),
                (result.right_taper, (255, 145, 85)),
            ):
                if taper is not None and settings.show_fit_line and taper.fit_line:
                    x1, y1, x2, y2 = taper.fit_line
                    draw.line([point(x1, y1), point(x2, y2)], fill=(20, 20, 26), width=3)
                    draw.line([point(x1, y1), point(x2, y2)], fill=color, width=2)
            ellipse = result.ellipse_cd
            if ellipse is not None and ellipse.center_x is not None and ellipse.center_y is not None:
                if ellipse.horizontal_diameter_px is not None and ellipse.vertical_diameter_px is not None:
                    cx, cy = point(ellipse.center_x, ellipse.center_y)
                    rx = int(round(ellipse.horizontal_diameter_px * scale_x * 0.5))
                    ry = int(round(ellipse.vertical_diameter_px * scale_y * 0.5))
                    draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=(125, 205, 255), width=2)
                for x, y in ellipse.boundary_points:
                    px, py = point(x, y)
                    draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=(245, 250, 255))
        return thumb

    def on_profile_hover(self, x: Optional[int], y: Optional[int]) -> None:
        self.profile_graph.update_cursor(x, y)

    def refresh_thumbnail_panel(self) -> None:
        self.thumbnail_panel.refresh(
            self.image_items,
            self.current_index,
            self.resolve_settings_for_item,
            self.get_thumbnail_preview,
        )

    def refresh_result_table(self) -> None:
        return

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
            settings.edge_scan_mode,
            settings.normalize_grayscale_profiles,
            settings.denoise_grayscale_profiles,
            settings.profile_denoise_window,
            round(float(settings.profile_denoise_range_sigma), 4),
            round(float(settings.normalize_low_percentile), 4),
            round(float(settings.normalize_high_percentile), 4),
            round(float(settings.normalize_min_span), 4),
            int(settings.max_profile_candidates_per_scanline),
            round(float(settings.minimum_grayscale_delta), 4),
            round(float(settings.max_jump_px), 4),
            bool(getattr(settings, "filter_cd_thk_by_boundary_angle", False)),
            round(float(getattr(settings, "max_cd_thk_boundary_angle_deg", 18.0)), 4),
            int(getattr(settings, "cd_thk_boundary_angle_window", 9)),
            round(float(settings.base_height_pct), 4),
            round(float(settings.left_offset_pct), 4),
            round(float(settings.right_offset_pct), 4),
            round(float(settings.taper_residual_limit_px), 4),
            settings.taper_left_edge_direction,
            settings.taper_right_edge_direction,
            round(float(cal.px_to_real), 8),
            cal.unit,
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
            self.render_current_image(update_option_settings=False)
        else:
            self.render_current_image(update_option_settings=False)

    def on_roi_changed(self, roi) -> None:
        item = self.current_item()
        if item is None:
            return
        clean_roi = normalize_roi(roi, item.image_size)
        if clean_roi is None:
            self.set_status(t(self.language, "roi_too_small"))
            return
        settings = self._ensure_item_settings(item, "image_specific")
        settings.roi = clean_roi
        settings.roi_source_image = item.file_name
        item.result = None
        self.set_status(t(self.language, "roi_applied"))
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
            settings = self._ensure_item_settings(item, "image_specific")
            settings.calibration.detected_scale_bar_px = float(pixel_length)
            settings.calibration.mode = "auto"
            self.option_panel.set_detected_scale_bar(float(pixel_length))
            self.scale_bar_bboxes[item.image_path] = result.get("bbox")
            self.set_status(t(self.language, "scale_bar_detected").format(pixel_length=pixel_length))
        else:
            settings = self._ensure_item_settings(item, "image_specific")
            settings.calibration.detected_scale_bar_px = None
            self.option_panel.set_detected_scale_bar(None)
            self.set_status(str(result.get("message", t(self.language, "scale_bar_failed"))))
            messagebox.showinfo(t(self.language, "scale_bar_detection"), str(result.get("message", t(self.language, "scale_bar_failed"))))
        self.render_current_image()

    def _targets_for_scope(self, scope: str) -> List[ImageItem]:
        item = self.current_item()
        if scope == "all":
            return list(self.image_items)
        return [item] if item else []

    def _apply_current_settings_to_targets(self, targets: List[ImageItem]) -> int:
        current = self.current_item()
        if current is None:
            return 0
        current_settings = self.resolve_settings_for_item(current)
        if current_settings.roi is None:
            return 0

        applied = 0
        for item in targets:
            settings = current_settings.clone()
            roi = normalize_roi(current_settings.roi, item.image_size)
            if roi is None:
                continue
            settings.roi = roi
            settings.roi_source_image = current.file_name
            settings.settings_source = "image_specific"
            item.settings = settings
            item.result = None
            applied += 1
        if applied:
            self.render_cache.clear()
            self.thumbnail_overlay_cache.clear()
        return applied

    def apply_calibration_to_scope(self) -> None:
        mode, pixel_length, actual_length, unit = self.option_panel.get_calibration_inputs()
        calibration = apply_calibration(pixel_length, actual_length, unit, mode=mode)
        if calibration.status != "calibrated":
            messagebox.showwarning(t(self.language, "calibration_failed"), t(self.language, "calibration_failed_message"))
            return
        item = self.current_item()
        if item is None:
            self.global_settings.calibration = calibration
            self.set_status(t(self.language, "default_calibration_applied").format(scale=calibration.px_to_real, unit=unit))
        else:
            settings = self._ensure_item_settings(item, "image_specific")
            settings.calibration = calibration
            self.set_status(
                t(self.language, "image_calibration_applied").format(
                    file_name=item.file_name,
                    scale=calibration.px_to_real,
                    unit=unit,
                )
            )
        self.render_current_image()

    def measure_scope(self, force_scope: Optional[str] = None) -> None:
        if not self.image_items:
            return
        scope = force_scope or "current"
        targets = self._targets_for_scope(scope)
        if not targets:
            return
        applied_roi_count = 0
        if scope == "all":
            applied_roi_count = self._apply_current_settings_to_targets(targets)
        failures = 0
        for idx, item in enumerate(targets, start=1):
            self.set_status(t(self.language, "measuring").format(index=idx, total=len(targets), file_name=item.file_name))
            image = self.load_image_cached(item.image_path)
            settings = self.resolve_settings_for_item(item)
            item.result = run_measurement(image, settings)
            if item.result.status == "Fail":
                failures += 1
        self.load_current_image()
        self.refresh_thumbnail_panel()
        self.refresh_result_table()
        roi_message = t(self.language, "roi_applied_count").format(count=applied_roi_count) if applied_roi_count else ""
        failure_message = t(self.language, "failures_count").format(count=failures) if failures else ""
        self.set_status(t(self.language, "measurement_complete").format(count=len(targets)) + roi_message + failure_message)

    def export_csv(self) -> None:
        if not self.image_items:
            return
        path = filedialog.asksaveasfilename(
            title="CSV 저장",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        export_results_to_csv(path, self.image_items, self.global_settings)
        self.set_status(t(self.language, "csv_saved").format(path=path))
        messagebox.showinfo(t(self.language, "csv_saved_title"), t(self.language, "csv_saved_message"))
