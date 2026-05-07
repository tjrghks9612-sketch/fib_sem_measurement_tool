from __future__ import annotations

import tkinter as tk
from typing import Callable, List

import customtkinter as ctk

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MEASUREMENT_TYPES, MeasurementSettings


ALL_TYPES = "전체 유형"
ALL_STATUSES = "전체 상태"
STATUS_LABELS = {
    "OK": "정상",
    "Check": "확인",
    "Review Needed": "검토 필요",
    "Fail": "실패",
    "Not measured": "측정 전",
}
STATUS_BY_LABEL = {label: status for status, label in STATUS_LABELS.items()}


class ThumbnailPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_select_image: Callable[[int], None],
        on_selection_changed: Callable[[], None],
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_select_image = on_select_image
        self.on_selection_changed = on_selection_changed
        self.items: List[ImageItem] = []
        self.current_index = -1
        self.resolve_settings: Callable[[ImageItem], MeasurementSettings] = lambda item: MeasurementSettings()
        self.image_refs = []
        self.visible_indices: List[int] = []

        self.type_filter = tk.StringVar(value=ALL_TYPES)
        self.status_filter = tk.StringVar(value=ALL_STATUSES)
        self.count_var = tk.StringVar(value="0개 이미지")
        self.selection_var = tk.StringVar(value="0 / 0")
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkLabel(header, text="이미지 / 결과", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, textvariable=self.count_var, text_color="#90a2b4").pack(side="right")

        filters = ctk.CTkFrame(self, fg_color="transparent")
        filters.pack(fill="x", padx=10, pady=(0, 6))
        self.type_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.type_filter,
            values=[ALL_TYPES] + list(MEASUREMENT_TYPES.values()),
            width=150,
            command=lambda _v: self._refresh_cards(),
        )
        self.type_menu.pack(side="left", padx=(0, 4))
        self.status_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.status_filter,
            values=[ALL_STATUSES] + list(STATUS_LABELS.values()),
            width=150,
            command=lambda _v: self._refresh_cards(),
        )
        self.status_menu.pack(side="left", padx=4)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#0a121b")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        footer = ctk.CTkFrame(self, fg_color="#0a121b")
        footer.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(footer, text="전체 선택", width=90, fg_color="#142234", command=self.select_all_visible).pack(
            side="left", padx=6, pady=6
        )
        ctk.CTkButton(footer, text="해제", width=90, fg_color="#142234", command=self.clear_selection).pack(
            side="left", padx=4, pady=6
        )
        ctk.CTkLabel(footer, text="선택").pack(side="left", padx=(14, 2))
        ctk.CTkLabel(footer, textvariable=self.selection_var, text_color="#48aaff").pack(side="left")

    def refresh(
        self,
        items: List[ImageItem],
        current_index: int,
        resolve_settings: Callable[[ImageItem], MeasurementSettings],
    ) -> None:
        self.items = items
        self.current_index = current_index
        self.resolve_settings = resolve_settings
        self._refresh_cards()

    def _passes_filter(self, item: ImageItem, settings: MeasurementSettings) -> bool:
        type_filter = self.type_filter.get()
        if type_filter != ALL_TYPES and MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type) != type_filter:
            return False
        status_filter = self.status_filter.get()
        status = item.result.status if item.result else "Not measured"
        selected_status = STATUS_BY_LABEL.get(status_filter, status_filter)
        if status_filter != ALL_STATUSES and status != selected_status:
            return False
        return True

    def _refresh_cards(self) -> None:
        for child in self.scroll.winfo_children():
            child.destroy()
        self.image_refs = []
        self.visible_indices = []

        for index, item in enumerate(self.items):
            settings = self.resolve_settings(item)
            if not self._passes_filter(item, settings):
                continue
            self.visible_indices.append(index)
            self._create_card(index, item, settings)

        self.count_var.set(f"{len(self.visible_indices)} / {len(self.items)}개 이미지")
        selected_count = sum(1 for item in self.items if item.selected)
        self.selection_var.set(f"{selected_count} / {len(self.items)}")

    def _status_color(self, status: str) -> str:
        return {
            "OK": "#52f36d",
            "Check": "#ffd452",
            "Review Needed": "#ff9f43",
            "Fail": "#ff5b69",
            "Not measured": "#8293a6",
        }.get(status, "#8293a6")

    def _create_card(self, index: int, item: ImageItem, settings: MeasurementSettings) -> None:
        selected = index == self.current_index
        border = "#0a84ff" if selected else "#213243"
        card = ctk.CTkFrame(
            self.scroll,
            fg_color="#0f1a25",
            border_color=border,
            border_width=2 if selected else 1,
            corner_radius=6,
        )
        card.pack(fill="x", pady=5)
        card.grid_columnconfigure(2, weight=1)
        var = tk.BooleanVar(value=item.selected)
        check = ctk.CTkCheckBox(card, text="", width=22, variable=var, command=lambda i=index, v=var: self._toggle(i, v))
        check.grid(row=0, column=0, rowspan=5, padx=(8, 4), pady=8, sticky="ns")

        if item.thumbnail is not None:
            image = ctk.CTkImage(light_image=item.thumbnail, dark_image=item.thumbnail, size=(124, 80))
            self.image_refs.append(image)
            thumb = ctk.CTkLabel(card, text="", image=image)
        else:
            thumb = ctk.CTkLabel(card, text="미리보기 없음", width=124, height=80)
        thumb.grid(row=0, column=1, rowspan=5, padx=6, pady=8)

        number = ctk.CTkLabel(card, text=f"{index + 1:02d}", fg_color="#164c94", corner_radius=4, width=28)
        number.grid(row=0, column=2, sticky="w", padx=(2, 0), pady=(8, 0))
        name = ctk.CTkLabel(card, text=item.file_name, anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        name.grid(row=0, column=2, sticky="ew", padx=(36, 8), pady=(8, 0))

        type_text = MEASUREMENT_TYPES.get(settings.measurement_type, settings.measurement_type)
        roi_text = "ROI 있음" if settings.roi else "ROI 없음"
        source_text = {
            "global_default": "기본 설정",
            "image_specific": "이미지별 설정",
        }.get(settings.settings_source, settings.settings_source)
        ctk.CTkLabel(card, text=f"{type_text} | {roi_text} | {source_text}", anchor="w", text_color="#c7d2df").grid(
            row=2, column=2, sticky="ew", padx=2
        )
        calibration_text = {
            "not_calibrated": "미보정",
            "calibrated": "보정됨",
        }.get(settings.calibration.status, settings.calibration.status)
        ctk.CTkLabel(card, text=f"캘리브레이션: {calibration_text}", anchor="w", text_color="#8ea0b1").grid(
            row=3, column=2, sticky="ew", padx=2
        )

        status = item.result.status if item.result else "Not measured"
        summary = item.result.compact_summary(settings.calibration.unit, settings.calibration.px_to_real) if item.result else "측정 전"
        ctk.CTkLabel(card, text=summary, anchor="w", text_color=self._status_color(status)).grid(
            row=4, column=2, sticky="ew", padx=2, pady=(0, 8)
        )

        self._bind_card_selection(card, index)

    def _bind_card_selection(self, widget, index: int) -> None:
        widget.bind("<Button-1>", lambda _event, i=index: self.on_select_image(i), add="+")
        for child in widget.winfo_children():
            self._bind_card_selection(child, index)

    def _toggle(self, index: int, var: tk.BooleanVar) -> None:
        self.items[index].selected = bool(var.get())
        self.on_selection_changed()
        self._refresh_cards()

    def select_all_visible(self) -> None:
        for index in self.visible_indices:
            self.items[index].selected = True
        self.on_selection_changed()
        self._refresh_cards()

    def clear_selection(self) -> None:
        for item in self.items:
            item.selected = False
        self.on_selection_changed()
        self._refresh_cards()
