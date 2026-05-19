from __future__ import annotations

import tkinter as tk
from typing import Callable, List, Optional

import customtkinter as ctk

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import measurement_key, measurement_label, settings_source_label, status_label, t


MEASUREMENT_KEYS = (
    "taper_single",
    "taper_double",
    "distance_horizontal",
    "distance_vertical",
    "distance_both",
    "hole_cd",
    "crater",
)
STATUS_KEYS = ("OK", "Check", "Review Needed", "Fail", "Not measured")


class ThumbnailPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_select_image: Callable[[int], None],
        on_selection_changed: Callable[[], None],
        on_delete_selected: Callable[[], None],
        on_measure_selected: Callable[[], None],
        language: str = "ko",
        **kwargs,
    ):
        super().__init__(master, fg_color="#0d1721", border_color="#223242", border_width=1, corner_radius=6, **kwargs)
        self.on_select_image = on_select_image
        self.on_selection_changed = on_selection_changed
        self.on_delete_selected = on_delete_selected
        self.on_measure_selected = on_measure_selected
        self.language = language
        self.items: List[ImageItem] = []
        self.current_index = -1
        self.resolve_settings: Callable[[ImageItem], MeasurementSettings] = lambda item: MeasurementSettings()
        self.render_thumbnail: Callable[[ImageItem, MeasurementSettings], Optional[object]] = lambda item, settings: item.thumbnail
        self.image_refs = []
        self.visible_indices: List[int] = []
        self.card_refs = {}

        self.type_filter = tk.StringVar(value=t(self.language, "all_types"))
        self.status_filter = tk.StringVar(value=t(self.language, "all_statuses"))
        self.count_var = tk.StringVar(value=f"0 {t(self.language, 'images_unit')}")
        self.selection_var = tk.StringVar(value="0 / 0")
        self._build()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 4))
        self.header_label = ctk.CTkLabel(header, text=t(self.language, "thumb_header"), font=ctk.CTkFont(size=15, weight="bold"))
        self.header_label.pack(side="left")
        ctk.CTkLabel(header, textvariable=self.count_var, text_color="#90a2b4").pack(side="right")

        filters = ctk.CTkFrame(self, fg_color="transparent")
        filters.pack(fill="x", padx=10, pady=(0, 6))
        self.type_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.type_filter,
            values=self._type_filter_values(),
            width=150,
            command=lambda _v: self._refresh_cards(),
        )
        self.type_menu.pack(side="left", padx=(0, 4))
        self.status_menu = ctk.CTkOptionMenu(
            filters,
            variable=self.status_filter,
            values=self._status_filter_values(),
            width=150,
            command=lambda _v: self._refresh_cards(),
        )
        self.status_menu.pack(side="left", padx=4)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="#0a121b")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        footer = ctk.CTkFrame(self, fg_color="#0a121b")
        footer.pack(fill="x", padx=10, pady=(0, 10))
        footer_actions = ctk.CTkFrame(footer, fg_color="transparent")
        footer_actions.pack(fill="x", padx=6, pady=(6, 3))
        footer_batch = ctk.CTkFrame(footer, fg_color="transparent")
        footer_batch.pack(fill="x", padx=6, pady=(3, 6))
        self.select_all_button = ctk.CTkButton(
            footer_actions,
            text=t(self.language, "select_all"),
            width=90,
            fg_color="#142234",
            command=self.select_all_visible,
        )
        self.select_all_button.pack(side="left", padx=(0, 4))
        self.clear_button = ctk.CTkButton(footer_actions, text=t(self.language, "clear"), width=90, fg_color="#142234", command=self.clear_selection)
        self.clear_button.pack(side="left", padx=4)
        self.selection_label = ctk.CTkLabel(footer_actions, text=t(self.language, "selection"))
        self.selection_label.pack(side="left", padx=(14, 2))
        ctk.CTkLabel(footer_actions, textvariable=self.selection_var, text_color="#48aaff").pack(side="left")
        self.delete_selected_button = ctk.CTkButton(
            footer_batch,
            text=t(self.language, "delete_selected_images"),
            width=128,
            fg_color="#462032",
            command=self.on_delete_selected,
        )
        self.delete_selected_button.pack(side="left", padx=(0, 4))
        self.measure_selected_button = ctk.CTkButton(
            footer_batch,
            text=t(self.language, "measure_selected"),
            width=128,
            command=self.on_measure_selected,
        )
        self.measure_selected_button.pack(side="left", padx=4)

    def _type_filter_values(self) -> list[str]:
        return [t(self.language, "all_types")] + [measurement_label(self.language, key) for key in MEASUREMENT_KEYS]

    def _status_filter_values(self) -> list[str]:
        return [t(self.language, "all_statuses")] + [status_label(self.language, key) for key in STATUS_KEYS]

    def refresh(
        self,
        items: List[ImageItem],
        current_index: int,
        resolve_settings: Callable[[ImageItem], MeasurementSettings],
        render_thumbnail: Optional[Callable[[ImageItem, MeasurementSettings], Optional[object]]] = None,
    ) -> None:
        self.items = items
        self.current_index = current_index
        self.resolve_settings = resolve_settings
        if render_thumbnail is not None:
            self.render_thumbnail = render_thumbnail
        self._refresh_cards()

    def _passes_filter(self, item: ImageItem, settings: MeasurementSettings) -> bool:
        type_filter = self.type_filter.get()
        if type_filter != t(self.language, "all_types") and settings.measurement_type != measurement_key(type_filter, ""):
            return False
        status_filter = self.status_filter.get()
        status = item.result.status if item.result else "Not measured"
        selected_status = next((key for key in STATUS_KEYS if status_label(self.language, key) == status_filter), status_filter)
        if status_filter != t(self.language, "all_statuses") and status != selected_status:
            return False
        return True

    def _refresh_cards(self) -> None:
        for child in self.scroll.winfo_children():
            child.destroy()
        self.image_refs = []
        self.visible_indices = []
        self.card_refs = {}

        for index, item in enumerate(self.items):
            settings = self.resolve_settings(item)
            if not self._passes_filter(item, settings):
                continue
            self.visible_indices.append(index)
            self._create_card(index, item, settings)

        self._update_counts()

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
        refs = {"frame": card}
        self.card_refs[index] = refs
        card.grid_columnconfigure(2, weight=1)
        var = tk.BooleanVar(value=item.selected)
        refs["check_var"] = var
        check = ctk.CTkCheckBox(card, text="", width=22, variable=var, command=lambda i=index, v=var: self._toggle(i, v))
        refs["check"] = check
        check.grid(row=0, column=0, rowspan=5, padx=(8, 4), pady=8, sticky="ns")

        preview = self.render_thumbnail(item, settings)
        if preview is not None:
            image = ctk.CTkImage(light_image=preview, dark_image=preview, size=(124, 80))
            self.image_refs.append(image)
            thumb = ctk.CTkLabel(card, text="", image=image)
        else:
            thumb = ctk.CTkLabel(card, text=t(self.language, "no_preview"), width=124, height=80)
        refs["thumb"] = thumb
        thumb.grid(row=0, column=1, rowspan=5, padx=6, pady=8)

        number = ctk.CTkLabel(card, text=f"{index + 1:02d}", fg_color="#164c94", corner_radius=4, width=28)
        refs["number"] = number
        number.grid(row=0, column=2, sticky="w", padx=(2, 0), pady=(8, 0))
        name = ctk.CTkLabel(card, text=item.file_name, anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        refs["name"] = name
        name.grid(row=0, column=2, sticky="ew", padx=(36, 8), pady=(8, 0))

        type_text = measurement_label(self.language, settings.measurement_type)
        roi_text = t(self.language, "roi_exists") if settings.roi else t(self.language, "roi_missing")
        source_text = settings_source_label(self.language, settings.settings_source)
        meta_label = ctk.CTkLabel(card, text=f"{type_text} | {roi_text} | {source_text}", anchor="w", text_color="#c7d2df")
        refs["meta"] = meta_label
        meta_label.grid(row=2, column=2, sticky="ew", padx=2)
        calibration_text = {
            "not_calibrated": t(self.language, "not_calibrated"),
            "calibrated": t(self.language, "calibrated"),
        }.get(settings.calibration.status, settings.calibration.status)
        calibration_label = ctk.CTkLabel(card, text=f"{t(self.language, 'calibration')}: {calibration_text}", anchor="w", text_color="#8ea0b1")
        refs["calibration"] = calibration_label
        calibration_label.grid(row=3, column=2, sticky="ew", padx=2)

        status = item.result.status if item.result else "Not measured"
        summary = item.result.compact_summary(settings.calibration.unit, settings.calibration.px_to_real) if item.result else t(self.language, "before_measurement")
        summary_label = ctk.CTkLabel(card, text=summary, anchor="w", text_color=self._status_color(status))
        refs["summary"] = summary_label
        summary_label.grid(row=4, column=2, sticky="ew", padx=2, pady=(0, 8))

        self._bind_card_selection(card, index)

    def set_current_index(self, current_index: int) -> None:
        if current_index == self.current_index:
            return
        previous = self.current_index
        self.current_index = current_index
        for index in (previous, current_index):
            refs = self.card_refs.get(index)
            if refs is None:
                continue
            card = refs["frame"]
            selected = index == current_index
            card.configure(
                border_color="#0a84ff" if selected else "#213243",
                border_width=2 if selected else 1,
            )

    def update_item(self, index: int) -> None:
        if not (0 <= index < len(self.items)):
            return
        item = self.items[index]
        settings = self.resolve_settings(item)
        passes_filter = self._passes_filter(item, settings)
        is_visible = index in self.card_refs
        if passes_filter != is_visible:
            self._refresh_cards()
            return
        if not passes_filter:
            self._update_counts()
            return

        refs = self.card_refs[index]
        card = refs["frame"]
        selected = index == self.current_index
        card.configure(
            border_color="#0a84ff" if selected else "#213243",
            border_width=2 if selected else 1,
        )
        refs["check_var"].set(bool(item.selected))
        refs["number"].configure(text=f"{index + 1:02d}")
        refs["name"].configure(text=item.file_name)

        preview = self.render_thumbnail(item, settings)
        if preview is not None:
            image = ctk.CTkImage(light_image=preview, dark_image=preview, size=(124, 80))
            self.image_refs.append(image)
            refs["thumb"].configure(text="", image=image)
        else:
            refs["thumb"].configure(text=t(self.language, "no_preview"), image=None)

        type_text = measurement_label(self.language, settings.measurement_type)
        roi_text = t(self.language, "roi_exists") if settings.roi else t(self.language, "roi_missing")
        source_text = settings_source_label(self.language, settings.settings_source)
        refs["meta"].configure(text=f"{type_text} | {roi_text} | {source_text}")

        calibration_text = {
            "not_calibrated": t(self.language, "not_calibrated"),
            "calibrated": t(self.language, "calibrated"),
        }.get(settings.calibration.status, settings.calibration.status)
        refs["calibration"].configure(text=f"{t(self.language, 'calibration')}: {calibration_text}")

        status = item.result.status if item.result else "Not measured"
        summary = item.result.compact_summary(settings.calibration.unit, settings.calibration.px_to_real) if item.result else t(self.language, "before_measurement")
        refs["summary"].configure(text=summary, text_color=self._status_color(status))
        self._update_counts()

    def update_current_card(self) -> None:
        self.update_item(self.current_index)

    def _update_counts(self) -> None:
        self.count_var.set(f"{len(self.visible_indices)} / {len(self.items)} {t(self.language, 'images_unit')}")
        selected_count = sum(1 for item in self.items if item.selected)
        self.selection_var.set(f"{selected_count} / {len(self.items)}")

    def _bind_card_selection(self, widget, index: int) -> None:
        widget.bind("<Button-1>", lambda _event, i=index: self.on_select_image(i), add="+")
        for child in widget.winfo_children():
            self._bind_card_selection(child, index)

    def _toggle(self, index: int, var: tk.BooleanVar) -> None:
        self.items[index].selected = bool(var.get())
        self.on_selection_changed()
        selected_count = sum(1 for item in self.items if item.selected)
        self.selection_var.set(f"{selected_count} / {len(self.items)}")

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

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        type_key = measurement_key(self.type_filter.get(), "")
        status_key = next((key for key in STATUS_KEYS if status_label(self.language, key) == self.status_filter.get()), "")
        self.language = language
        self.header_label.configure(text=t(self.language, "thumb_header"))
        self.select_all_button.configure(text=t(self.language, "select_all"))
        self.clear_button.configure(text=t(self.language, "clear"))
        self.delete_selected_button.configure(text=t(self.language, "delete_selected_images"))
        self.measure_selected_button.configure(text=t(self.language, "measure_selected"))
        self.selection_label.configure(text=t(self.language, "selection"))
        self.type_menu.configure(values=self._type_filter_values())
        self.status_menu.configure(values=self._status_filter_values())
        self.type_filter.set(measurement_label(self.language, type_key) if type_key else t(self.language, "all_types"))
        self.status_filter.set(status_label(self.language, status_key) if status_key else t(self.language, "all_statuses"))
        self._refresh_cards()
