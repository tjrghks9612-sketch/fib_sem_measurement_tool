from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import customtkinter as ctk
import numpy as np

from fib_sem_measurement_tool.core.grayscale_line_scan import (
    prepare_display_profile_from_roi,
    prepare_display_profile_signal,
)
from fib_sem_measurement_tool.core.profile_markers import collect_profile_edge_markers
from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import profile_mode_key, profile_mode_label, t


MARKER_COLORS = {
    "CD": "#ffa437",
    "THK": "#50f578",
    "taper": "#ffdc4b",
}


class ProfileGraph(ctk.CTkFrame):
    def __init__(self, master, language: str = "ko", **kwargs):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.language = language
        self.mode_var = tk.StringVar(value=profile_mode_label(self.language, "both"))
        self.coord_var = tk.StringVar(value=t(self.language, "profile_hover_prompt"))
        self.gray: Optional[np.ndarray] = None
        self.result: Optional[MeasurementResult] = None
        self.settings = MeasurementSettings()
        self.cursor: Optional[Tuple[int, int]] = None
        self._image_id = None
        self._result_id = None
        self._settings_signature = None
        self._pending_cursor: Optional[Tuple[Optional[int], Optional[int]]] = None
        self._draw_after_id = None
        self._last_drawn_cursor: Optional[Tuple[int, int]] = None
        self._suspend_draw = False
        self._draw_dirty = False
        self._build()

    def _request_draw(self) -> None:
        if self._suspend_draw:
            self._draw_dirty = True
            return
        self._draw_graph()

    def set_context(
        self,
        image_bgr,
        settings: MeasurementSettings,
        result: Optional[MeasurementResult],
    ) -> None:
        self._suspend_draw = True
        self._draw_dirty = False
        try:
            self.set_image(image_bgr)
            self.set_settings(settings)
            self.set_result(result)
        finally:
            self._suspend_draw = False
        if self._draw_dirty:
            self._draw_dirty = False
            self._draw_graph()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))
        self.title_label = ctk.CTkLabel(header, text=t(self.language, "profile_title"), font=ctk.CTkFont(size=13, weight="bold"))
        self.title_label.pack(side="left")
        ctk.CTkLabel(header, textvariable=self.coord_var, text_color="#90a2b4").pack(side="left", padx=12)
        self.mode = ctk.CTkSegmentedButton(
            header,
            values=[profile_mode_label(self.language, key) for key in ("both", "horizontal", "vertical")],
            variable=self.mode_var,
            command=lambda _value: self._draw_graph(),
            width=220,
        )
        self.mode.pack(side="right")

        self.canvas = tk.Canvas(self, bg="#071018", height=210, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._draw_graph())

    def set_image(self, image_bgr) -> None:
        if image_bgr is None:
            self.gray = None
            self.result = None
            self.cursor = None
            self._image_id = None
            self._result_id = None
            self._last_drawn_cursor = None
            self.coord_var.set(t(self.language, "profile_no_image"))
            self._request_draw()
            return
        image_id = (id(image_bgr), image_bgr.shape)
        if image_id == self._image_id:
            return
        if image_bgr.ndim == 2:
            self.gray = image_bgr.copy()
        else:
            self.gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self._image_id = image_id
        self.cursor = None
        self._last_drawn_cursor = None
        self.coord_var.set(t(self.language, "profile_hover_prompt"))
        self._request_draw()

    def set_result(self, result: Optional[MeasurementResult]) -> None:
        result_id = id(result) if result is not None else None
        if result_id == self._result_id:
            return
        self.result = result
        self._result_id = result_id
        self._request_draw()

    def set_settings(self, settings: MeasurementSettings) -> None:
        signature = (
            settings.roi,
            bool(getattr(settings, "normalize_grayscale_profiles", False)),
            bool(getattr(settings, "denoise_grayscale_profiles", False)),
            int(getattr(settings, "profile_denoise_window", 3)),
            round(float(getattr(settings, "profile_denoise_range_sigma", 28.0)), 4),
            round(float(getattr(settings, "normalize_low_percentile", 2.0)), 4),
            round(float(getattr(settings, "normalize_high_percentile", 98.0)), 4),
            round(float(getattr(settings, "normalize_min_span", 12.0)), 4),
        )
        if signature == self._settings_signature:
            return
        self.settings = settings.clone()
        self._settings_signature = signature
        self._last_drawn_cursor = None
        self._request_draw()

    def clear_cursor(self) -> None:
        if self.cursor is None and self._pending_cursor == (None, None):
            return
        self.cursor = None
        self._last_drawn_cursor = None
        self.coord_var.set(t(self.language, "profile_hover_prompt"))
        self._draw_graph()

    def update_cursor(self, x: Optional[int], y: Optional[int]) -> None:
        if self.gray is None or x is None or y is None:
            self.clear_cursor()
            return
        height, width = self.gray.shape[:2]
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        if self._last_drawn_cursor == (x, y):
            return
        self._pending_cursor = (x, y)
        if self._draw_after_id is None:
            self._draw_after_id = self.after(35, self._flush_cursor_update)

    def _flush_cursor_update(self) -> None:
        self._draw_after_id = None
        if self._pending_cursor is None:
            return
        x, y = self._pending_cursor
        self._pending_cursor = None
        if self.gray is None or x is None or y is None:
            self.clear_cursor()
            return
        self.cursor = (x, y)
        self._last_drawn_cursor = self.cursor
        mode_text = []
        if bool(getattr(self.settings, "normalize_grayscale_profiles", False)):
            mode_text.append(t(self.language, "profile_normalized"))
        if bool(getattr(self.settings, "denoise_grayscale_profiles", False)):
            mode_text.append(t(self.language, "profile_smoothed"))
        basis = t(self.language, "profile_roi") if self.settings.roi is not None else t(self.language, "profile_full")
        suffix = f" / {t(self.language, 'profile_graph_basis')}: {basis}"
        if mode_text:
            suffix += f" {'+'.join(mode_text)}"
        self.coord_var.set(f"x={x}, y={y}, {t(self.language, 'profile_original')}={int(self.gray[y, x])}{suffix}")
        self._draw_graph()

    def _profile_points(self, profile: np.ndarray, width: int, height: int):
        if profile.size <= 1:
            return []
        xs = np.linspace(0, width - 1, profile.size)
        ys = height - 1 - (profile.astype(np.float32) / 255.0) * (height - 16) - 8
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def _draw_polyline(self, points, color: str, step: int = 1) -> None:
        if len(points) < 2:
            return
        if step > 1:
            points = points[::step]
        flat = []
        for x, y in points:
            flat.extend([x, y])
        self.canvas.create_line(*flat, fill=color, width=2, smooth=True)

    def _marker_color(self, label: str) -> str:
        for prefix, color in MARKER_COLORS.items():
            if label.startswith(prefix) or prefix in label:
                return color
        return "#f7e36d"

    def _profile_x(self, position: float, profile_size: int, width: int) -> float:
        if profile_size <= 1:
            return 0.0
        clamped = max(0.0, min(float(profile_size - 1), float(position)))
        return clamped / float(profile_size - 1) * float(width - 1)

    def _profile_y(self, value: float, height: int) -> float:
        return float(height - 1 - (float(value) / 255.0) * (height - 16) - 8)

    def _draw_edge_marker(self, profile: np.ndarray, position: float, label: str, width: int, height: int, row: int) -> None:
        if profile.size == 0:
            return
        color = self._marker_color(label)
        x = self._profile_x(position, int(profile.size), width)
        index = int(round(max(0.0, min(float(profile.size - 1), float(position)))))
        y = self._profile_y(float(profile[index]), height)
        self.canvas.create_line(x, 8, x, height - 8, fill=color, dash=(3, 4), width=1)
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, fill="#071018", width=2)
        anchor = "nw" if x < width - 72 else "ne"
        text_x = x + 5 if anchor == "nw" else x - 5
        text_y = min(height - 20, max(10, y + (row % 3 - 1) * 16))
        self.canvas.create_text(text_x, text_y, text=self._marker_label(label), fill=color, anchor=anchor, font=("Arial", 8, "bold"))

    def _marker_label(self, label: str) -> str:
        side_left = t(self.language, "left_taper").replace(t(self.language, "taper"), "").strip()
        side_right = t(self.language, "right_taper").replace(t(self.language, "taper"), "").strip()
        return {
            "CD L": f"CD {side_left}",
            "CD R": f"CD {side_right}",
            "THK T": "THK T",
            "THK B": "THK B",
            "L taper": t(self.language, "left_taper"),
            "R taper": t(self.language, "right_taper"),
        }.get(label, label)

    def _draw_edge_markers(
        self,
        axis: str,
        scan_index: int,
        profile: np.ndarray,
        width: int,
        height: int,
        position_offset: int = 0,
    ) -> None:
        for idx, marker in enumerate(collect_profile_edge_markers(self.result, axis, scan_index)):
            self._draw_edge_marker(profile, marker.position - position_offset, marker.label, width, height, idx)

    def _profile_for_axis(self, axis: str, scan_index: int) -> tuple[np.ndarray, int, int]:
        if self.gray is None:
            return np.asarray([], dtype=np.float32), 0, scan_index
        if self.settings.roi is not None:
            return prepare_display_profile_from_roi(self.gray, self.settings.roi, axis, scan_index, self.settings)
        if axis == "horizontal":
            return prepare_display_profile_signal(self.gray[scan_index, :], "horizontal", self.settings), 0, scan_index
        return prepare_display_profile_signal(self.gray[:, scan_index], "vertical", self.settings), 0, scan_index

    def _draw_graph(self) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, width, height, fill="#071018", outline="")
        for value in (64, 128, 192):
            y = height - 1 - (value / 255.0) * (height - 16) - 8
            self.canvas.create_line(0, y, width, y, fill="#142536")
            self.canvas.create_text(4, y - 2, text=str(value), fill="#647487", anchor="sw", font=("Arial", 8))

        if self.gray is None:
            self.canvas.create_text(width / 2, height / 2, text=t(self.language, "profile_load_image"), fill="#8293a6")
            return
        if self.cursor is None:
            self.canvas.create_text(width / 2, height / 2, text=t(self.language, "profile_hover_main"), fill="#8293a6")
            return

        x, y = self.cursor
        mode = profile_mode_key(self.mode_var.get())
        if mode in ("both", "horizontal"):
            horizontal, offset, scan_index = self._profile_for_axis("horizontal", y)
            step = max(1, int(len(horizontal) / max(width, 1)))
            self._draw_polyline(self._profile_points(horizontal, width, height), "#38a8ff", step)
            self._draw_edge_markers("horizontal", scan_index, horizontal, width, height, offset)
            self.canvas.create_text(width - 8, 14, text=profile_mode_label(self.language, "horizontal"), fill="#38a8ff", anchor="ne", font=("Arial", 10, "bold"))
        if mode in ("both", "vertical"):
            vertical, offset, scan_index = self._profile_for_axis("vertical", x)
            step = max(1, int(len(vertical) / max(width, 1)))
            self._draw_polyline(self._profile_points(vertical, width, height), "#61f27a", step)
            self._draw_edge_markers("vertical", scan_index, vertical, width, height, offset)
            self.canvas.create_text(width - 8, 30, text=profile_mode_label(self.language, "vertical"), fill="#61f27a", anchor="ne", font=("Arial", 10, "bold"))

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        mode_key = profile_mode_key(self.mode_var.get())
        self.language = language
        self.title_label.configure(text=t(self.language, "profile_title"))
        values = [profile_mode_label(self.language, key) for key in ("both", "horizontal", "vertical")]
        self.mode.configure(values=values)
        self.mode_var.set(profile_mode_label(self.language, mode_key))
        if self.gray is None:
            self.coord_var.set(t(self.language, "profile_no_image"))
        elif self.cursor is None:
            self.coord_var.set(t(self.language, "profile_hover_prompt"))
        self._draw_graph()
