from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional, Tuple

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from fib_sem_measurement_tool.ui.i18n import t


class ImageViewer(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_roi_changed: Callable[[Tuple[int, int, int, int]], None],
        on_overlay_toggled: Callable[[bool], None],
        on_hover_profile: Optional[Callable[[Optional[int], Optional[int]], None]] = None,
        on_manual_points: Optional[Callable[[list[Tuple[int, int]]], None]] = None,
        on_clear_measurement: Optional[Callable[[], None]] = None,
        language: str = "ko",
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_roi_changed = on_roi_changed
        self.on_overlay_toggled = on_overlay_toggled
        self.on_hover_profile = on_hover_profile
        self.on_manual_points = on_manual_points
        self.on_clear_measurement = on_clear_measurement
        self.language = language

        self.title_var = tk.StringVar(value=t(self.language, "viewer_title"))
        self.meta_var = tk.StringVar(value=t(self.language, "load_image_prompt"))
        self.status_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value=t(self.language, "fit"))
        self.mode_var = tk.StringVar(value="roi")
        self.overlay_enabled = True

        self.image_bgr = None
        self.render_bgr = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.fit_mode = True
        self.zoom = 1.0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.draw_w = 0
        self.draw_h = 0
        self.drag_start_canvas: Optional[Tuple[int, int]] = None
        self.drag_item: Optional[int] = None
        self.manual_points: list[Tuple[int, int]] = []
        self.manual_point_count = 2
        self.manual_measurement_type = ""
        self.manual_items: list[int] = []
        self.manual_hint_items: list[int] = []
        self._content_image_id = None
        self._last_render_state = None
        self._last_hover_xy = (None, None)

        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(header, textvariable=self.title_var, font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, textvariable=self.status_var, text_color="#52f36d").pack(side="right")

        ctk.CTkLabel(self, textvariable=self.meta_var, anchor="w", text_color="#aebccb").pack(fill="x", padx=12, pady=(0, 6))

        self.canvas = tk.Canvas(self, bg="#070d13", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        controls = ctk.CTkFrame(self, fg_color="#0a121b")
        controls.pack(fill="x", padx=12, pady=(0, 10))
        primary_button_width = 76
        self.roi_button = ctk.CTkButton(controls, text="ROI", width=primary_button_width, command=lambda: self.set_mode("roi"))
        self.roi_button.pack(side="left", padx=(8, 4), pady=8)
        self.manual_button = ctk.CTkButton(
            controls,
            text=t(self.language, "manual_mode"),
            width=primary_button_width,
            fg_color="#142234",
            command=lambda: self.set_mode("manual"),
        )
        self.manual_button.pack(side="left", padx=4, pady=8)
        self.manual_clear_button = ctk.CTkButton(
            controls,
            text=t(self.language, "manual_clear"),
            width=primary_button_width,
            fg_color="#142234",
            command=self.clear_measurement_marks,
        )
        self.manual_clear_button.pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="-", width=38, fg_color="#142234", command=self.zoom_out).pack(side="left", padx=4, pady=8)
        ctk.CTkButton(controls, text="+", width=38, fg_color="#142234", command=self.zoom_in).pack(side="left", padx=4, pady=8)
        self.fit_button = ctk.CTkButton(controls, text=t(self.language, "fit"), width=54, fg_color="#142234", command=self.fit)
        self.fit_button.pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(controls, textvariable=self.zoom_var, width=70).pack(side="left", padx=(8, 4))
        self.overlay_switch = ctk.CTkSwitch(controls, text=t(self.language, "overlay"), command=self._toggle_overlay)
        self.overlay_switch.select()
        self.overlay_switch.pack(side="right", padx=10, pady=8)

    def set_mode(self, mode: str) -> None:
        clean_mode = "manual" if mode == "manual" else "roi"
        self.mode_var.set(clean_mode)
        self.canvas.configure(cursor="tcross" if clean_mode == "manual" else "crosshair")
        self.roi_button.configure(fg_color="#1f6aa5" if clean_mode == "roi" else "#142234")
        self.manual_button.configure(fg_color="#1f6aa5" if clean_mode == "manual" else "#142234")
        if clean_mode != "manual":
            self._clear_manual_hint()
        self._update_manual_status()

    def set_manual_point_count(self, count: int, measurement_type: str = "") -> None:
        next_count = max(2, int(count))
        next_type = measurement_type or self.manual_measurement_type
        if next_count == self.manual_point_count and next_type == self.manual_measurement_type:
            return
        self.manual_point_count = next_count
        self.manual_measurement_type = next_type
        self.clear_manual_points()

    def set_content(self, image_bgr, render_bgr, title: str, meta: str, status: str) -> None:
        image_id = (id(image_bgr), image_bgr.shape if image_bgr is not None else None)
        self.image_bgr = image_bgr
        self.render_bgr = render_bgr
        self.title_var.set(title)
        self.meta_var.set(meta)
        self.status_var.set(status)
        if image_id != self._content_image_id:
            self.fit_mode = True
            self._content_image_id = image_id
        self._render()

    def clear(self) -> None:
        self.image_bgr = None
        self.render_bgr = None
        self.canvas.delete("all")
        self._canvas_image_item = None
        self._last_render_state = None
        self._last_hover_xy = (None, None)
        self.clear_manual_points()
        self.title_var.set(t(self.language, "viewer_title"))
        self.meta_var.set(t(self.language, "load_image_prompt"))
        self.status_var.set("")

    def _toggle_overlay(self) -> None:
        self.overlay_enabled = bool(self.overlay_switch.get())
        self.on_overlay_toggled(self.overlay_enabled)

    def zoom_in(self) -> None:
        self.fit_mode = False
        self.zoom = min(8.0, self.zoom * 1.25)
        self._render(preserve_offset=False)

    def zoom_out(self) -> None:
        self.fit_mode = False
        self.zoom = max(0.05, self.zoom / 1.25)
        self._render(preserve_offset=False)

    def fit(self) -> None:
        self.fit_mode = True
        self._render()

    def _on_mouse_wheel(self, event) -> None:
        factor = 1.25 if event.delta > 0 else 1 / 1.25
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, canvas_x: int, canvas_y: int, factor: float) -> None:
        if self.render_bgr is None:
            return
        if self.draw_w <= 0 or self.draw_h <= 0:
            self._render()
        old_scale = max(self.scale, 1e-6)
        if not self._is_canvas_inside_image(canvas_x, canvas_y):
            canvas_x = self.canvas.winfo_width() // 2
            canvas_y = self.canvas.winfo_height() // 2
        image_x = (canvas_x - self.offset_x) / old_scale
        image_y = (canvas_y - self.offset_y) / old_scale
        self.fit_mode = False
        self.zoom = max(0.05, min(8.0, self.zoom * factor))
        self.scale = self.zoom
        self.offset_x = int(round(canvas_x - image_x * self.scale))
        self.offset_y = int(round(canvas_y - image_y * self.scale))
        self._render(preserve_offset=True)

    def _render(self, preserve_offset: bool = False) -> None:
        if self.render_bgr is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        h, w = self.render_bgr.shape[:2]
        if self.fit_mode:
            self.scale = min(canvas_w / w, canvas_h / h)
            self.zoom = self.scale
        else:
            self.scale = self.zoom
        self.draw_w = max(1, int(w * self.scale))
        self.draw_h = max(1, int(h * self.scale))
        if self.fit_mode or not preserve_offset:
            self.offset_x = int((canvas_w - self.draw_w) / 2)
            self.offset_y = int((canvas_h - self.draw_h) / 2)
        else:
            self._clamp_offsets(canvas_w, canvas_h)
        render_state = (id(self.render_bgr), self.draw_w, self.draw_h, self.offset_x, self.offset_y)
        if render_state != self._last_render_state:
            rgb = cv2.cvtColor(self.render_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((self.draw_w, self.draw_h), Image.BILINEAR)
            self.tk_image = ImageTk.PhotoImage(pil)
            self._last_render_state = render_state
        if getattr(self, "_canvas_image_item", None) is None:
            self._canvas_image_item = self.canvas.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor="nw")
        else:
            self.canvas.coords(self._canvas_image_item, self.offset_x, self.offset_y)
            self.canvas.itemconfig(self._canvas_image_item, image=self.tk_image)
        if self.fit_mode:
            self.zoom_var.set(t(self.language, "fit"))
        else:
            self.zoom_var.set(f"{self.scale * 100:.0f}%")
        self._redraw_manual_points()

    def _clamp_offsets(self, canvas_w: int, canvas_h: int) -> None:
        if self.draw_w <= canvas_w:
            self.offset_x = int((canvas_w - self.draw_w) / 2)
        else:
            self.offset_x = max(canvas_w - self.draw_w, min(0, int(self.offset_x)))
        if self.draw_h <= canvas_h:
            self.offset_y = int((canvas_h - self.draw_h) / 2)
        else:
            self.offset_y = max(canvas_h - self.draw_h, min(0, int(self.offset_y)))

    def _update_manual_status(self) -> None:
        if self.mode_var.get() == "manual":
            self.status_var.set(t(self.language, "manual_points_status").format(count=len(self.manual_points), total=self.manual_point_count))

    def clear_manual_points(self) -> None:
        self.manual_points = []
        for item in self.manual_items:
            self.canvas.delete(item)
        self.manual_items = []
        self._clear_manual_hint()
        self._update_manual_status()

    def clear_measurement_marks(self) -> None:
        self.clear_manual_points()
        if self.drag_item:
            self.canvas.delete(self.drag_item)
            self.drag_item = None
        self.drag_start_canvas = None
        if self.on_clear_measurement is not None:
            self.on_clear_measurement()

    def _image_to_canvas(self, point: Tuple[int, int]) -> Tuple[int, int]:
        return int(round(self.offset_x + point[0] * self.scale)), int(round(self.offset_y + point[1] * self.scale))

    def _redraw_manual_points(self) -> None:
        for item in self.manual_items:
            self.canvas.delete(item)
        self.manual_items = []
        if not self.manual_points:
            return
        canvas_points = [self._image_to_canvas(point) for point in self.manual_points]
        for p1, p2 in self._manual_line_segments(canvas_points):
            self.manual_items.append(self.canvas.create_line(*p1, *p2, fill="#ffde59", width=2, dash=(5, 3)))
        for idx, (x, y) in enumerate(canvas_points, start=1):
            self.manual_items.append(self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#101820", fill="#ffde59", width=2))
            self.manual_items.append(self.canvas.create_text(x + 10, y - 10, text=str(idx), fill="#fffbdf", anchor="w"))

    def _clear_manual_hint(self) -> None:
        for item in self.manual_hint_items:
            self.canvas.delete(item)
        self.manual_hint_items = []

    def _manual_hint_text(self) -> str:
        step = min(len(self.manual_points) + 1, self.manual_point_count)
        mode = self.manual_measurement_type
        labels = {
            "distance_horizontal": ("1. 왼쪽 경계", "2. 오른쪽 경계"),
            "distance_vertical": ("1. 위쪽 경계", "2. 아래쪽 경계"),
            "distance_both": ("1. 가로 시작", "2. 가로 끝", "3. 세로 시작", "4. 세로 끝"),
            "hole_cd": ("1. 가로 시작", "2. 가로 끝", "3. 세로 시작", "4. 세로 끝"),
            "crater": ("1. 왼쪽 foot", "2. 오른쪽 foot", "3. 상단 높이"),
            "taper_single": ("1. 시작점", "2. 끝점"),
            "taper_double": ("1. 좌측 시작", "2. 좌측 끝", "3. 우측 시작", "4. 우측 끝"),
        }.get(mode, ("1. 시작점", "2. 끝점"))
        return labels[step - 1] if 0 <= step - 1 < len(labels) else f"{step}. 점 선택"

    def _update_manual_hint(self, x: int, y: int) -> None:
        self._clear_manual_hint()
        if self.mode_var.get() != "manual" or self.image_bgr is None or not self._is_canvas_inside_image(x, y):
            return
        text = self._manual_hint_text()
        pad_x, pad_y = 8, 5
        text_id = self.canvas.create_text(0, 0, text=text, fill="#eef6ff", font=("Malgun Gothic", 10), anchor="nw")
        bbox = self.canvas.bbox(text_id)
        if bbox is None:
            self.canvas.delete(text_id)
            return
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        left = min(max(x + 18, 4), max(4, self.canvas.winfo_width() - tw - pad_x * 2 - 4))
        top = min(max(y + 12, 4), max(4, self.canvas.winfo_height() - th - pad_y * 2 - 4))
        self.canvas.coords(text_id, left + pad_x, top + pad_y)
        rect_id = self.canvas.create_rectangle(
            left,
            top,
            left + tw + pad_x * 2,
            top + th + pad_y * 2,
            fill="#101820",
            outline="#2f4154",
            width=1,
        )
        self.canvas.tag_raise(text_id, rect_id)
        self.manual_hint_items = [rect_id, text_id]

    def _manual_line_segments(self, canvas_points: list[Tuple[int, int]]) -> list[tuple[Tuple[int, int], Tuple[int, int]]]:
        mode = self.manual_measurement_type
        segments: list[tuple[Tuple[int, int], Tuple[int, int]]] = []

        def horizontal(p1: Tuple[int, int], p2: Tuple[int, int]) -> tuple[Tuple[int, int], Tuple[int, int]]:
            y = int(round((p1[1] + p2[1]) * 0.5))
            return (p1[0], y), (p2[0], y)

        def vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> tuple[Tuple[int, int], Tuple[int, int]]:
            x = int(round((p1[0] + p2[0]) * 0.5))
            return (x, p1[1]), (x, p2[1])

        if len(canvas_points) >= 2:
            if mode == "distance_horizontal":
                segments.append(horizontal(canvas_points[0], canvas_points[1]))
            elif mode == "distance_vertical":
                segments.append(vertical(canvas_points[0], canvas_points[1]))
            elif mode in {"distance_both", "hole_cd", "crater"}:
                segments.append(horizontal(canvas_points[0], canvas_points[1]))
            else:
                segments.append((canvas_points[0], canvas_points[1]))
        if mode == "crater" and len(canvas_points) >= 3:
            baseline_y = int(round((canvas_points[0][1] + canvas_points[1][1]) * 0.5))
            top = canvas_points[2]
            segments.append(vertical(top, (top[0], baseline_y)))
        if len(canvas_points) >= 4:
            if mode in {"distance_both", "hole_cd"}:
                segments.append(vertical(canvas_points[2], canvas_points[3]))
            elif mode == "taper_double":
                segments.append((canvas_points[2], canvas_points[3]))
        return segments

    def _is_canvas_inside_image(self, x: int, y: int) -> bool:
        return (
            self.image_bgr is not None
            and self.offset_x <= x < self.offset_x + self.draw_w
            and self.offset_y <= y < self.offset_y + self.draw_h
        )

    def _canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        if self.image_bgr is None:
            return 0, 0
        h, w = self.image_bgr.shape[:2]
        ix = int(round((x - self.offset_x) / max(self.scale, 1e-6)))
        iy = int(round((y - self.offset_y) / max(self.scale, 1e-6)))
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def _on_motion(self, event) -> None:
        self._update_manual_hint(event.x, event.y)
        if self.on_hover_profile is None:
            return
        if not self._is_canvas_inside_image(event.x, event.y):
            if self._last_hover_xy != (None, None):
                self._last_hover_xy = (None, None)
                self.on_hover_profile(None, None)
            return
        point = self._canvas_to_image(event.x, event.y)
        if point != self._last_hover_xy:
            self._last_hover_xy = point
            self.on_hover_profile(*point)

    def _on_leave(self, _event) -> None:
        self._clear_manual_hint()
        if self.on_hover_profile is not None and self._last_hover_xy != (None, None):
            self._last_hover_xy = (None, None)
            self.on_hover_profile(None, None)

    def _on_press(self, event) -> None:
        if self.image_bgr is None:
            return
        if self.mode_var.get() == "manual":
            if not self._is_canvas_inside_image(event.x, event.y):
                return
            self.manual_points.append(self._canvas_to_image(event.x, event.y))
            self._redraw_manual_points()
            self._update_manual_status()
            if len(self.manual_points) >= self.manual_point_count and self.on_manual_points is not None:
                points = list(self.manual_points[: self.manual_point_count])
                self.clear_manual_points()
                self.on_manual_points(points)
            else:
                self._update_manual_hint(event.x, event.y)
            return
        self.drag_start_canvas = (event.x, event.y)
        if self.drag_item:
            self.canvas.delete(self.drag_item)
            self.drag_item = None

    def _on_drag(self, event) -> None:
        if self.drag_start_canvas is None:
            return
        if self.drag_item:
            self.canvas.delete(self.drag_item)
        x0, y0 = self.drag_start_canvas
        self.drag_item = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="#ffd23c", dash=(4, 3), width=2)

    def _on_release(self, event) -> None:
        if self.drag_start_canvas is None:
            return
        start = self._canvas_to_image(*self.drag_start_canvas)
        end = self._canvas_to_image(event.x, event.y)
        self.drag_start_canvas = None
        if self.drag_item:
            self.canvas.delete(self.drag_item)
            self.drag_item = None

        x1, y1 = start
        x2, y2 = end
        if abs(x2 - x1) >= 8 and abs(y2 - y1) >= 8:
            self.on_roi_changed((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        self.language = language
        self.fit_button.configure(text=t(self.language, "fit"))
        self.overlay_switch.configure(text=t(self.language, "overlay"))
        if self.image_bgr is None:
            self.title_var.set(t(self.language, "viewer_title"))
            self.meta_var.set(t(self.language, "load_image_prompt"))
        if self.fit_mode:
            self.zoom_var.set(t(self.language, "fit"))
        self.manual_button.configure(text=t(self.language, "manual_mode"))
        self.manual_clear_button.configure(text=t(self.language, "manual_clear"))
        self._update_manual_status()
