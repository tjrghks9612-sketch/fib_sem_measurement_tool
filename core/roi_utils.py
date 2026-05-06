from __future__ import annotations

from typing import Optional, Sequence, Tuple

Roi = Tuple[int, int, int, int]


def normalize_roi(roi: Sequence[float], image_size: Tuple[int, int]) -> Optional[Roi]:
    if not roi or len(roi) != 4:
        return None
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2

