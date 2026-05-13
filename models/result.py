from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MeasurementStatus(str, Enum):
    OK = "OK"
    CHECK = "Check"
    REVIEW_NEEDED = "Review Needed"
    FAIL = "Fail"


@dataclass(frozen=True)
class RawEdgeCandidate:
    scan_axis: str
    scan_index: int
    local_scan_index: int
    position: float
    image_x: float
    image_y: float
    strength: float
    signed_delta: float
    sign: int
    grayscale_before: float
    grayscale_after: float


@dataclass(frozen=True)
class PairCandidate:
    scan_axis: str
    scan_index: int
    local_scan_index: int
    first: RawEdgeCandidate
    second: RawEdgeCandidate

    @property
    def distance(self) -> float:
        return float(self.second.position - self.first.position)


@dataclass
class EdgeScanResult:
    scan_axis: str
    roi: Tuple[int, int, int, int]
    scanned_line_count: int
    raw_edge_candidates: List[RawEdgeCandidate] = field(default_factory=list)
    per_scanline_candidate_count: Dict[int, int] = field(default_factory=dict)
    profiles_by_scanline: Dict[int, List[float]] = field(default_factory=dict)
    gradient_projection: List[float] = field(default_factory=list)
    minimum_grayscale_delta: float = 0.0

    @property
    def raw_edge_count(self) -> int:
        return len(self.raw_edge_candidates)

    @property
    def valid_scanline_count(self) -> int:
        return sum(1 for count in self.per_scanline_candidate_count.values() if count > 0)

    @property
    def scanline_coverage(self) -> float:
        if self.scanned_line_count <= 0:
            return 0.0
        return float(self.valid_scanline_count / self.scanned_line_count)

    @property
    def raw_edge_density(self) -> float:
        if self.scanned_line_count <= 0:
            return 0.0
        return float(self.raw_edge_count / self.scanned_line_count)


@dataclass
class DistanceResult:
    orientation: str
    mean_px: Optional[float] = None
    max_px: Optional[float] = None
    min_px: Optional[float] = None
    median_px: Optional[float] = None
    std_px: Optional[float] = None
    selected_px: Optional[float] = None
    selected_method: str = "mean"
    valid_count: int = 0
    total_count: int = 0
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    boundary_pairs: List[Tuple[float, float, float]] = field(default_factory=list)
    values_px: List[float] = field(default_factory=list)
    raw_edge_candidates: List[RawEdgeCandidate] = field(default_factory=list)
    pair_candidates: List[PairCandidate] = field(default_factory=list)
    selected_pairs: List[PairCandidate] = field(default_factory=list)
    selected_pair: Optional[PairCandidate] = None
    per_scanline_candidate_count: Dict[int, int] = field(default_factory=dict)
    raw_edge_count: int = 0
    pair_candidate_count: int = 0
    scanned_line_count: int = 0
    valid_scanline_count: int = 0
    scanline_coverage: float = 0.0
    selected_point_count: int = 0
    raw_edge_density: float = 0.0
    minimum_grayscale_delta: float = 0.0

    def scaled(self, px_to_real: float) -> Dict[str, Optional[float]]:
        if not px_to_real:
            px_to_real = 1.0
        return {
            "mean": self.mean_px * px_to_real if self.mean_px is not None else None,
            "max": self.max_px * px_to_real if self.max_px is not None else None,
            "min": self.min_px * px_to_real if self.min_px is not None else None,
            "median": self.median_px * px_to_real if self.median_px is not None else None,
            "std": self.std_px * px_to_real if self.std_px is not None else None,
            "selected": self.selected_px * px_to_real if self.selected_px is not None else None,
        }


@dataclass
class TaperSideResult:
    side: str
    angle_horizontal: Optional[float] = None
    angle_vertical: Optional[float] = None
    fit_r2: Optional[float] = None
    fit_error: Optional[float] = None
    valid_point_count: int = 0
    inlier_count: int = 0
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    points: List[Tuple[float, float]] = field(default_factory=list)
    fit_line: Optional[Tuple[float, float, float, float]] = None
    raw_edge_candidates: List[RawEdgeCandidate] = field(default_factory=list)
    selected_boundary_candidates: List[RawEdgeCandidate] = field(default_factory=list)
    per_scanline_candidate_count: Dict[int, int] = field(default_factory=dict)
    raw_edge_count: int = 0
    scanned_line_count: int = 0
    valid_scanline_count: int = 0
    scanline_coverage: float = 0.0
    selected_point_count: int = 0
    fit_point_count: int = 0
    raw_edge_density: float = 0.0
    minimum_grayscale_delta: float = 0.0


@dataclass
class EllipseCDResult:
    ray_attempt_count: int = 16
    valid_point_count: int = 0
    outlier_count: int = 0
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    major_axis_px: Optional[float] = None
    minor_axis_px: Optional[float] = None
    angle_deg: Optional[float] = None
    horizontal_diameter_px: Optional[float] = None
    vertical_diameter_px: Optional[float] = None
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    boundary_points: List[Tuple[float, float]] = field(default_factory=list)
    edge_strength_mean: Optional[float] = None
    radius_cv: Optional[float] = None

    def scaled(self, px_to_real: float) -> Dict[str, Optional[float]]:
        if not px_to_real:
            px_to_real = 1.0
        return {
            "horizontal_diameter": self.horizontal_diameter_px * px_to_real
            if self.horizontal_diameter_px is not None
            else None,
            "vertical_diameter": self.vertical_diameter_px * px_to_real
            if self.vertical_diameter_px is not None
            else None,
        }


@dataclass
class HoleCDResult:
    target: str = "inner"
    horizontal_px: Optional[float] = None
    vertical_px: Optional[float] = None
    min_feret_px: Optional[float] = None
    max_feret_px: Optional[float] = None
    equivalent_diameter_px: Optional[float] = None
    area_px: Optional[float] = None
    perimeter_px: Optional[float] = None
    coverage: float = 0.0
    mean_radius: Optional[float] = None
    radius_std: Optional[float] = None
    mean_strength: Optional[float] = None
    smoothness: Optional[float] = None
    continuity: float = 0.0
    gap_count: int = 0
    max_gap: int = 0
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    contour_points: List[Tuple[float, float]] = field(default_factory=list)
    center: Optional[Tuple[float, float]] = None
    ellipse_major_px: Optional[float] = None
    ellipse_minor_px: Optional[float] = None
    ellipse_angle_deg: Optional[float] = None
    ellipse_fit_error: Optional[float] = None
    ellipse_aspect_ratio: Optional[float] = None

    def scaled(self, px_to_real: float) -> Dict[str, Optional[float]]:
        if not px_to_real:
            px_to_real = 1.0
        return {
            "horizontal": self.horizontal_px * px_to_real if self.horizontal_px is not None else None,
            "vertical": self.vertical_px * px_to_real if self.vertical_px is not None else None,
            "min_feret": self.min_feret_px * px_to_real if self.min_feret_px is not None else None,
            "max_feret": self.max_feret_px * px_to_real if self.max_feret_px is not None else None,
            "equivalent_diameter": self.equivalent_diameter_px * px_to_real if self.equivalent_diameter_px is not None else None,
        }


@dataclass
class MeasurementResult:
    measurement_type: str
    horizontal_cd: Optional[DistanceResult] = None
    vertical_thk: Optional[DistanceResult] = None
    left_taper: Optional[TaperSideResult] = None
    right_taper: Optional[TaperSideResult] = None
    ellipse_cd: Optional[EllipseCDResult] = None
    hole_cd: Optional[HoleCDResult] = None
    avg_taper_angle: Optional[float] = None
    taper_angle_diff: Optional[float] = None
    overall_confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    measurement_source: str = "auto"

    def compact_summary(self, unit: str, px_to_real: float) -> str:
        chunks: List[str] = []
        if self.horizontal_cd and self.horizontal_cd.selected_px is not None:
            value = self.horizontal_cd.selected_px * px_to_real
            chunks.append(f"CD {value:.3g} {unit}")
        if self.vertical_thk and self.vertical_thk.selected_px is not None:
            value = self.vertical_thk.selected_px * px_to_real
            chunks.append(f"THK {value:.3g} {unit}")
        if self.left_taper and self.left_taper.angle_horizontal is not None:
            chunks.append(f"좌 {self.left_taper.angle_horizontal:.1f} deg")
        if self.right_taper and self.right_taper.angle_horizontal is not None:
            chunks.append(f"우 {self.right_taper.angle_horizontal:.1f} deg")
        if self.ellipse_cd and self.ellipse_cd.horizontal_diameter_px is not None:
            h_value = self.ellipse_cd.horizontal_diameter_px * px_to_real
            v_value = (
                self.ellipse_cd.vertical_diameter_px * px_to_real
                if self.ellipse_cd.vertical_diameter_px is not None
                else None
            )
            chunks.append(f"Ellipse H {h_value:.3g} {unit}")
            if v_value is not None:
                chunks.append(f"Ellipse V {v_value:.3g} {unit}")
        if self.hole_cd and self.hole_cd.horizontal_px is not None:
            h_value = self.hole_cd.horizontal_px * px_to_real
            v_value = self.hole_cd.vertical_px * px_to_real if self.hole_cd.vertical_px is not None else None
            chunks.append(f"Hole H {h_value:.3g} {unit}")
            if v_value is not None:
                chunks.append(f"Hole V {v_value:.3g} {unit}")
        status_label = {
            "OK": "정상",
            "Check": "확인",
            "Review Needed": "검토 필요",
            "Fail": "실패",
        }.get(self.status, self.status)
        chunks.append(f"{status_label} 신뢰도 {self.overall_confidence:.0f}%")
        return " | ".join(chunks)

    def raw_edge_count(self) -> int:
        return sum(
            item.raw_edge_count
            for item in (self.horizontal_cd, self.vertical_thk, self.left_taper, self.right_taper)
            if item is not None
        )

    def selected_point_count(self) -> int:
        count = sum(
            item.selected_point_count
            for item in (self.horizontal_cd, self.vertical_thk, self.left_taper, self.right_taper)
            if item is not None
        )
        if self.ellipse_cd is not None:
            count += self.ellipse_cd.valid_point_count
        if self.hole_cd is not None:
            count += len(self.hole_cd.contour_points)
        return count
