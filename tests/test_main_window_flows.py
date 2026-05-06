from __future__ import annotations

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui import main_window
from fib_sem_measurement_tool.ui.main_window import MainWindow


def test_add_image_paths_ignores_duplicates_within_single_call(monkeypatch) -> None:
    window = MainWindow.__new__(MainWindow)
    window.image_items = []
    window.current_index = 0
    window.load_current_image = lambda: None
    window.refresh_all = lambda: None
    window.set_status = lambda _message: None

    monkeypatch.setattr(main_window, "read_image_metadata", lambda _path: ((100, 80), None))

    window.add_image_paths(["C:/images/a.png", "C:/images/a.png"])

    assert [item.image_path for item in window.image_items] == ["C:/images/a.png"]


def test_calibration_line_updates_manual_pixel_input() -> None:
    class OptionPanelStub:
        def __init__(self) -> None:
            self.pixel_length = None

        def set_manual_calibration_length(self, pixel_length: float) -> None:
            self.pixel_length = pixel_length

    window = MainWindow.__new__(MainWindow)
    window.calibration_lines = {}
    window.option_panel = OptionPanelStub()
    window.set_status = lambda _message: None
    window.render_current_image = lambda: None
    window.current_item = lambda: ImageItem("C:/images/a.png", "a.png", (100, 80))

    window.on_calibration_line((0, 0, 3, 4), 5.0)

    assert window.calibration_lines["C:/images/a.png"] == (0, 0, 3, 4)
    assert window.option_panel.pixel_length == 5.0


def test_measure_scope_batches_work_with_after(monkeypatch) -> None:
    first = ImageItem("C:/images/a.png", "a.png", (100, 80))
    second = ImageItem("C:/images/b.png", "b.png", (100, 80))
    window = MainWindow.__new__(MainWindow)
    window.image_items = [first, second]
    window.current_index = 0
    window.group_settings = {}
    window.global_settings = MeasurementSettings(roi=(10, 10, 80, 70))
    window.calibration_lines = {}
    window.scale_bar_bboxes = {}
    window.load_image_cached = lambda path: path
    window.load_current_image = lambda: None
    window.refresh_thumbnail_panel = lambda: None
    window.set_status = lambda _message: None
    scheduled = []
    window.after = lambda delay, callback, *args: scheduled.append((delay, callback, args)) or "after-id"

    monkeypatch.setattr(
        main_window,
        "run_measurement",
        lambda _image, settings: MeasurementResult(measurement_type=settings.measurement_type, status="OK", overall_confidence=90.0),
    )

    window.measure_scope("all")

    assert first.result is not None
    assert second.result is None
    assert scheduled

    _delay, callback, args = scheduled.pop(0)
    callback(*args)

    assert second.result is not None


def test_measure_scope_records_item_error_and_continues(monkeypatch) -> None:
    first = ImageItem("C:/images/a.png", "a.png", (100, 80))
    second = ImageItem("C:/images/b.png", "b.png", (100, 80))
    window = MainWindow.__new__(MainWindow)
    window.image_items = [first, second]
    window.current_index = 0
    window.group_settings = {}
    window.global_settings = MeasurementSettings(roi=(10, 10, 80, 70))
    window.calibration_lines = {}
    window.scale_bar_bboxes = {}
    window.load_image_cached = lambda path: path
    window.load_current_image = lambda: None
    window.refresh_thumbnail_panel = lambda: None
    window.set_status = lambda _message: None
    scheduled = []
    window.after = lambda delay, callback, *args: scheduled.append((delay, callback, args)) or "after-id"

    def fake_run_measurement(image, settings):
        if image == "C:/images/a.png":
            raise ValueError("bad image")
        return MeasurementResult(measurement_type=settings.measurement_type, status="OK", overall_confidence=90.0)

    monkeypatch.setattr(main_window, "run_measurement", fake_run_measurement)

    window.measure_scope("all")

    assert first.result is not None
    assert first.result.status == "Fail"
    assert first.last_error == "bad image"
    assert scheduled

    _delay, callback, args = scheduled.pop(0)
    callback(*args)

    assert second.result is not None
    assert second.result.status == "OK"
