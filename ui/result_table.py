from __future__ import annotations

from typing import Callable, List

import customtkinter as ctk

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MEASUREMENT_TYPES, MeasurementSettings


class ResultTable(ctk.CTkFrame):
    COLUMNS = [
        ("File", 190),
        ("Mode", 130),
        ("ROI", 140),
        ("Value", 120),
        ("Unit", 56),
        ("Raw Edges", 82),
        ("Coverage", 78),
        ("Selected", 76),
        ("Threshold", 76),
        ("Status", 98),
    ]

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(header, text="Results", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        table_header = ctk.CTkFrame(self, fg_color="#0a121b", corner_radius=4)
        table_header.pack(fill="x", padx=10, pady=(0, 2))
        for col, (title, width) in enumerate(self.COLUMNS):
            table_header.grid_columnconfigure(col, minsize=width, weight=1 if col == 0 else 0)
            ctk.CTkLabel(table_header, text=title, text_color="#8ea0b1", anchor="w", font=ctk.CTkFont(size=11, weight="bold")).grid(
                row=0, column=col, sticky="ew", padx=6, pady=6
            )

        self.rows = ctk.CTkScrollableFrame(self, fg_color="transparent", height=132)
        self.rows.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def refresh(self, items: List[ImageItem], resolve_settings: Callable[[ImageItem], MeasurementSettings]) -> None:
        for child in self.rows.winfo_children():
            child.destroy()
        if not items:
            ctk.CTkLabel(self.rows, text="No images loaded.", text_color="#8293a6").pack(fill="x", pady=18)
            return
        for row_index, item in enumerate(items):
            self._row(row_index, item, resolve_settings(item))

    def _row(self, row_index: int, item: ImageItem, settings: MeasurementSettings) -> None:
        values = self._values(item, settings)
        frame = ctk.CTkFrame(self.rows, fg_color="#101b27" if row_index % 2 == 0 else "#0c1621", corner_radius=3)
        frame.pack(fill="x", pady=1)
        for col, ((_, width), value) in enumerate(zip(self.COLUMNS, values)):
            frame.grid_columnconfigure(col, minsize=width, weight=1 if col == 0 else 0)
            ctk.CTkLabel(frame, text=value, text_color=self._value_color(col, value), anchor="w").grid(
                row=0, column=col, sticky="ew", padx=6, pady=5
            )

    def _values(self, item: ImageItem, settings: MeasurementSettings) -> list[str]:
        result = item.result
        roi = "-" if settings.roi is None else f"{settings.roi[0]},{settings.roi[1]},{settings.roi[2]},{settings.roi[3]}"
        mode = MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type)
        if result is None:
            return [item.file_name, mode, roi, "-", settings.calibration.unit, "-", "-", "-", f"{settings.minimum_grayscale_delta:.0f}", "Not measured"]

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
            result.status,
        ]

    def _value_color(self, col: int, value: str) -> str:
        if col == 9:
            return {
                "OK": "#56f08a",
                "Check": "#ffd452",
                "Review Needed": "#ff9f43",
                "Fail": "#ff5b69",
                "Not measured": "#8293a6",
            }.get(value, "#dbe7f2")
        if col in {3, 5, 6, 7, 8}:
            return "#f4f8fb"
        return "#c7d2df"
