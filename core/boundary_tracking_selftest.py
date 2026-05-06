from __future__ import annotations

import numpy as np

from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_taper import measure_single_taper
from fib_sem_measurement_tool.core.preprocessing import preprocess_image
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def run_core_selftest() -> dict:
    img = np.zeros((120, 180), dtype=np.uint8)
    img[:, 50:55] = 220
    img[:, 120:128] = 220
    img[20:100, 80:85] = 140
    noisy = np.clip(img + (np.random.randn(*img.shape) * 14), 0, 255).astype(np.uint8)
    s = MeasurementSettings(roi=(10, 10, 170, 110), measurement_type="distance_both")
    s.apply_noise_preset(force=True)
    pre = preprocess_image(noisy, s)

    refs = {}
    for ref in ["outer", "inner", "center", "strongest"]:
        s.edge_reference = ref
        refs[ref] = measure_horizontal_cd(pre, s.roi, s).selected_px

    h = measure_horizontal_cd(pre, s.roi, s)
    v = measure_vertical_thk(pre, s.roi, s)
    t = measure_single_taper(pre, s.roi, "left", s)
    return {
        "edge_reference_diff": len(set([round(x or -1, 2) for x in refs.values()])) > 1,
        "horizontal_has_stats": all(getattr(h, k) is not None for k in ["mean_px", "max_px", "min_px", "median_px", "std_px"]),
        "vertical_has_stats": all(getattr(v, k) is not None for k in ["mean_px", "max_px", "min_px", "median_px", "std_px"]),
        "taper_single_works": (t.left_taper is not None and t.left_taper.status != "Fail"),
        "preprocess_applied": float(np.std(pre.astype(np.float32))) != float(np.std(noisy.astype(np.float32))),
    }
