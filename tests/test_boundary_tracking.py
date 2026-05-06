from __future__ import annotations

import numpy as np

from fib_sem_measurement_tool.core.boundary_tracking import build_boundary_track, extract_edge_bands
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def test_boundary_prior_selects_separate_left_and_right_edges() -> None:
    settings = MeasurementSettings(noise_level="low")
    settings.apply_noise_preset(force=True)
    settings.edge_reference = "inner"
    profile = np.zeros(161, dtype=np.float32)
    profile[50:94] = 220.0

    scan_bands = [
        extract_edge_bands(profile, scan_index, float(20 + scan_index * 10), 10.0, settings)
        for scan_index in range(6)
    ]

    left = build_boundary_track(scan_bands, "left", settings.edge_reference, settings)
    right = build_boundary_track(scan_bands, "right", settings.edge_reference, settings)

    assert left is not None
    assert right is not None
    left_x = float(np.median([point[1] for point in left.points]))
    right_x = float(np.median([point[1] for point in right.points]))
    assert left_x < 65.0
    assert right_x > 95.0
    assert right_x - left_x > 30.0
