from __future__ import annotations

from typing import Callable, List

import customtkinter as ctk

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import measurement_label, status_label, t


class ResultTable(ctk.CTkFrame):
    COLUMN_KEYS = [
        ("file", 190),
        ("measurement_mode", 130),
        ("ROI", 140),
        ("value", 120),
        ("unit", 56),
        ("raw_edge_count", 82),
        ("confidence", 78),
        ("selection", 76),
        ("threshold", 76),
        ("status", 98),
    ]

    def __init__(self, master, language: str = "ko", **kwargs):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.language = language
        self._items: List[ImageItem] = []
        self._resolve_settings: Callable[[ImageItem], MeasurementSettings] = lambda _item: MeasurementSettings()
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))
        self.header_label = ctk.CTkLabel(header, text=t(self.language, "section_selected_result"), font=ctk.CTkFont(size=14, weight="bold"))
        self.header_label.pack(side="left")

        self.table_header = ctk.CTkFrame(self, fg_color="#0a121b", corner_radius=4)
        self.table_header.pack(fill="x", padx=10, pady=(0, 2))
        self._build_header_columns()

        self.rows = ctk.CTkScrollableFrame(self, fg_color="transparent", height=132)
        self.rows.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def _column_label(self, key: str) -> str:
        return {
            "file": "File" if self.language == "en" else "Tệp" if self.language == "vi" else "파일",
            "value": "Value" if self.language == "en" else "Giá trị" if self.language == "vi" else "값",
            "status": "Status" if self.language == "en" else "Trạng thái" if self.language == "vi" else "상태",
        }.get(key, t(self.language, key))

    def _build_header_columns(self) -> None:
        for child in self.table_header.winfo_children():
            child.destroy()
        for col, (key, width) in enumerate(self.COLUMN_KEYS):
            self.table_header.grid_columnconfigure(col, minsize=width, weight=1 if col == 0 else 0)
            ctk.CTkLabel(self.table_header, text=self._column_label(key), text_color="#8ea0b1", anchor="w", font=ctk.CTkFont(size=11, weight="bold")).grid(
                row=0, column=col, sticky="ew", padx=6, pady=6
            )

    def refresh(self, items: List[ImageItem], resolve_settings: Callable[[ImageItem], MeasurementSettings]) -> None:
        self._items = items
        self._resolve_settings = resolve_settings
        for child in self.rows.winfo_children():
            child.destroy()
        if not items:
            ctk.CTkLabel(self.rows, text=t(self.language, "profile_load_image"), text_color="#8293a6").pack(fill="x", pady=18)
            return
        for row_index, item in enumerate(items):
            self._row(row_index, item, resolve_settings(item))

    def _row(self, row_index: int, item: ImageItem, settings: MeasurementSettings) -> None:
        values = self._values(item, settings)
        frame = ctk.CTkFrame(self.rows, fg_color="#101b27" if row_index % 2 == 0 else "#0c1621", corner_radius=3)
        frame.pack(fill="x", pady=1)
        for col, ((_, width), value) in enumerate(zip(self.COLUMN_KEYS, values)):
            frame.grid_columnconfigure(col, minsize=width, weight=1 if col == 0 else 0)
            ctk.CTkLabel(frame, text=value, text_color=self._value_color(col, value), anchor="w").grid(
                row=0, column=col, sticky="ew", padx=6, pady=5
            )

    def _values(self, item: ImageItem, settings: MeasurementSettings) -> list[str]:
        result = item.result
        roi = "-" if settings.roi is None else f"{settings.roi[0]},{settings.roi[1]},{settings.roi[2]},{settings.roi[3]}"
        mode = measurement_label(self.language, settings.measurement_type)
        if result is None:
            return [item.file_name, mode, roi, "-", settings.calibration.unit, "-", "-", "-", f"{settings.minimum_grayscale_delta:.0f}", t(self.language, "before_measurement")]

        value = "-"
        if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
            value = f"{result.horizontal_cd.selected_px * settings.calibration.px_to_real:.4g}"
        elif result.vertical_thk and result.vertical_thk.selected_px is not None:
            value = f"{result.vertical_thk.selected_px * settings.calibration.px_to_real:.4g}"
        elif result.avg_taper_angle is not None:
            value = f"{result.avg_taper_angle:.3g} deg"
        elif result.left_taper and result.left_taper.angle_horizontal is not None:
            value = f"{result.left_taper.angle_horizontal:.3g} deg"
        elif result.right_taper and result.right_taper.angle_horizontal is not None:
            value = f"{result.right_taper.angle_horizontal:.3g} deg"
        elif result.ellipse_cd and result.ellipse_cd.horizontal_diameter_px is not None:
            value = f"H {result.ellipse_cd.horizontal_diameter_px * settings.calibration.px_to_real:.4g}"

        return [
            item.file_name,
            mode,
            roi,
            value,
            settings.calibration.unit,
            str(result.raw_edge_count()),
            f"{result.overall_confidence:.0f}%",
            str(result.selected_point_count()),
            f"{settings.minimum_grayscale_delta:.0f}",
            self._status_label(result.status),
        ]

    def _status_label(self, status: str) -> str:
        return status_label(self.language, status)

    def _value_color(self, col: int, value: str) -> str:
        if col == 9:
            return {
                status_label(self.language, "OK"): "#56f08a",
                status_label(self.language, "Check"): "#ffd452",
                status_label(self.language, "Review Needed"): "#ff9f43",
                status_label(self.language, "Fail"): "#ff5b69",
                status_label(self.language, "Not measured"): "#8293a6",
            }.get(value, "#dbe7f2")
        if col in {3, 5, 6, 7, 8}:
            return "#f4f8fb"
        return "#c7d2df"

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        self.language = language
        self.header_label.configure(text=t(self.language, "section_selected_result"))
        self._build_header_columns()
        self.refresh(self._items, self._resolve_settings)
