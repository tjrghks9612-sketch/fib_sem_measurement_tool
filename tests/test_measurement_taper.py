from __future__ import annotations

from fib_sem_measurement_tool.core.boundary_tracking import BoundaryTrack
from fib_sem_measurement_tool.core.measurement_taper import _fit_track
from fib_sem_measurement_tool.models.result import MeasurementStatus
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def test_taper_fit_error_threshold_prevents_ok_status() -> None:
    settings = MeasurementSettings()
    settings.advanced.confidence_threshold = 40.0
    settings.advanced.fit_error_threshold = 2.0
    track = BoundaryTrack(
        points=[(0.0, 20.0), (10.0, 60.0), (20.0, 18.0), (30.0, 62.0), (40.0, 21.0), (50.0, 59.0)],
        side="left",
        coverage=1.0,
        mean_strength=20.0,
        smoothness=1.0,
        continuity=1.0,
        fit_error=0.0,
        score=100.0,
    )

    result = _fit_track(track, "left", settings)

    assert result.fit_error is not None
    assert result.fit_error > settings.advanced.fit_error_threshold
    assert result.status != MeasurementStatus.OK.value
    assert "fit error" in result.warning_message
