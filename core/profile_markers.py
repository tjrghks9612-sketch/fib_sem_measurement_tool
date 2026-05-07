from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from fib_sem_measurement_tool.models.result import MeasurementResult, PairCandidate, RawEdgeCandidate


@dataclass(frozen=True)
class ProfileEdgeMarker:
    axis: str
    position: float
    label: str


def _nearest_pair(pairs: list[PairCandidate], scan_index: int, tolerance: int) -> Optional[PairCandidate]:
    if not pairs:
        return None
    pair = min(pairs, key=lambda item: abs(int(item.scan_index) - int(scan_index)))
    if abs(int(pair.scan_index) - int(scan_index)) > int(tolerance):
        return None
    return pair


def _nearest_candidate(
    candidates: list[RawEdgeCandidate],
    scan_index: int,
    tolerance: int,
) -> Optional[RawEdgeCandidate]:
    if not candidates:
        return None
    candidate = min(candidates, key=lambda item: abs(int(item.scan_index) - int(scan_index)))
    if abs(int(candidate.scan_index) - int(scan_index)) > int(tolerance):
        return None
    return candidate


def collect_profile_edge_markers(
    result: Optional[MeasurementResult],
    axis: str,
    scan_index: int,
    tolerance: int = 3,
) -> List[ProfileEdgeMarker]:
    if result is None:
        return []

    markers: List[ProfileEdgeMarker] = []
    if axis == "horizontal":
        if result.horizontal_cd:
            pair = _nearest_pair(result.horizontal_cd.selected_pairs, scan_index, tolerance) or result.horizontal_cd.selected_pair
            if pair is not None:
                markers.append(ProfileEdgeMarker(axis=axis, position=pair.first.image_x, label="CD L"))
                markers.append(ProfileEdgeMarker(axis=axis, position=pair.second.image_x, label="CD R"))
        for taper in (result.left_taper, result.right_taper):
            if taper is None:
                continue
            candidate = _nearest_candidate(taper.selected_boundary_candidates, scan_index, tolerance)
            if candidate is not None:
                markers.append(ProfileEdgeMarker(axis=axis, position=candidate.image_x, label=f"{taper.side[0].upper()} taper"))
    elif axis == "vertical":
        if result.vertical_thk:
            pair = _nearest_pair(result.vertical_thk.selected_pairs, scan_index, tolerance) or result.vertical_thk.selected_pair
            if pair is not None:
                markers.append(ProfileEdgeMarker(axis=axis, position=pair.first.image_y, label="THK T"))
                markers.append(ProfileEdgeMarker(axis=axis, position=pair.second.image_y, label="THK B"))
    return markers
