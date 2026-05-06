from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.boundary_tracking import edge_position, extract_edge_bands
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def generate_scan_lines(roi: Sequence[int], direction: str, count: int, margin_ratio: float = 0.12):
    x1, y1, x2, y2 = [int(v) for v in roi]
    count = max(3, int(count))
    lines = []
    if direction == "horizontal":
        for y in np.linspace(y1 + (y2 - y1) * margin_ratio, y2 - (y2 - y1) * margin_ratio, count):
            lines.append(((x1, int(round(y))), (x2, int(round(y)))))
    else:
        for x in np.linspace(x1 + (x2 - x1) * margin_ratio, x2 - (x2 - x1) * margin_ratio, count):
            lines.append(((int(round(x)), y1), (int(round(x)), y2)))
    return lines


def detect_edge_candidates(profile: Sequence[float], settings: MeasurementSettings) -> List[Dict[str, float]]:
    bands = extract_edge_bands(profile, 0, 0.0, 0.0, settings)
    out = []
    for b in bands:
        out.append({"index": b.peak, "strength": b.strength, "signed_gradient": float(b.sign), "band_start": b.start, "band_end": b.end, "band_center": b.center, "band_peak": b.peak, "scan_index": b.scan_index})
    return out


def group_edge_bands(candidates: Iterable[Dict[str, float]], settings: MeasurementSettings) -> List[Dict[str, float]]:
    grouped = []
    for c in candidates:
        grouped.append({"start": c.get("band_start", c["index"]), "end": c.get("band_end", c["index"]), "center": c.get("band_center", c["index"]), "peak": c.get("band_peak", c["index"]), "strength": c["strength"], "signed_gradient": c.get("signed_gradient", 1.0)})
    return sorted(grouped, key=lambda x: x["center"])


def select_edge_from_band(edge_band: Dict[str, float], edge_reference: str, side: str) -> float:
    class B: pass
    b = B(); b.start = edge_band["start"]; b.end = edge_band["end"]; b.center = edge_band["center"]; b.peak = edge_band["peak"]
    return float(edge_position(b, edge_reference, side))


def find_boundary(profile: Sequence[float], side: str, settings: MeasurementSettings) -> Optional[float]:
    bands = group_edge_bands(detect_edge_candidates(profile, settings), settings)
    if not bands:
        return None
    center = len(profile) / 2.0
    filtered = [b for b in bands if (b["center"] <= center if side in {"left", "top"} else b["center"] >= center)] or bands
    pick = max(filtered, key=lambda b: b["strength"])
    return select_edge_from_band(pick, settings.edge_reference, side)


def find_boundary_pair(profile: Sequence[float], settings: MeasurementSettings) -> Optional[Tuple[float, float]]:
    left = find_boundary(profile, "left", settings); right = find_boundary(profile, "right", settings)
    if left is None or right is None or right <= left:
        return None
    return float(left), float(right)
