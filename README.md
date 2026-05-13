# FIB/SEM Measurement Tool

Desktop FIB/SEM metrology tool for measuring CD, THK, single taper, and double taper inside an ROI.

The current measurement pipeline is a Sobel-gradient boundary scanner. It preserves raw debug candidates while selecting the strongest local peak edges, using structural projection votes for layer selection, and smoothing selected boundary curves for more stable CD, THK, and taper measurement.

## Run

```powershell
python -m pip install -r fib_sem_measurement_tool\requirements.txt
python -m fib_sem_measurement_tool.main
```

or:

```powershell
python fib_sem_measurement_tool\main.py
```

## Measurement Algorithm

The measurement path is:

1. Input image
2. Convert to grayscale with `to_gray(image)`
3. Pass raw grayscale directly into measurement functions
4. Scan every row or every column inside the ROI
5. Calculate a direction-specific Sobel gradient (`dx=1` for horizontal scans, `dy=1` for vertical scans)
6. Emit local peak edge candidates that meet `minimum_grayscale_delta`, capped by `max_profile_candidates_per_scanline` so normalized noisy profiles cannot flood the pair selector
7. Repair selected boundary jumps, fill small NaN gaps, and smooth selected curves with a NaN-aware moving average

No threshold image, morphology, Canny, or image enhancement is applied to CD, THK, or taper measurement.

The inspector also exposes optional signal preparation controls. `Normalize Signal` applies robust percentile normalization to the ROI profile signal before Sobel detection, which helps low-contrast images. `Denoise Signal` applies a scan-axis median filter before Sobel detection. Both controls affect detection only; raw grayscale values remain preserved in candidate metadata and profile views.

## Raw Edge Candidate Definition

For each 1D grayscale profile, candidates come from the matching Sobel gradient profile. The Sobel output is normalized back to grayscale-delta units so existing threshold values remain interpretable:

```python
data = np.asarray(profile, dtype=np.float32)
gradient = cv2.Sobel(..., ksize=3) / 4.0
strength = abs(gradient)
position = local_peak_transition + 0.5
```

A candidate is emitted when:

```python
strength[local_peak] >= settings.minimum_grayscale_delta
```

Each candidate stores scan axis, original image scanline index, ROI-local scanline index, subpixel edge position, image coordinate, strength, signed delta, sign, and grayscale values before/after the jump.

## CD and THK

CD and THK are separated into two phases:

1. Generate all raw edge candidates for every ROI row or column.
2. Generate all same-scanline pair candidates for debug/export compatibility.
3. Select boundary pairs from Sobel local peaks, repair row/column jumps larger than `max_jump_px`, and post-process selected boundary curves with small-gap interpolation and NaN-aware moving average smoothing.
4. For wide vertical THK scans, first aggregate Sobel edge strength across the ROI into a projection profile and select the strongest separated horizontal layer votes. If projection coverage is insufficient, fall back to horizontal edge-density clusters and then generic pair selection.

The selected value remains available for compatibility, but the result also preserves:

- `raw_edge_candidates`
- `pair_candidates`
- `selected_pair`
- `selected_pairs`
- `per_scanline_candidate_count`
- `raw_edge_count`
- `pair_candidate_count`
- `scanline_coverage`
- `selected_point_count`

For `distance_both`, the horizontal CD pass runs first. THK then scans every column inside a CD-derived x range, using the median left/right CD boundaries plus a small internal margin instead of scanning the full ROI width.

## Taper

Single and double taper no longer depend on CD/THK pair selection.

Taper measurement first gathers horizontal Sobel local peak candidates inside the ROI. The left and right boundaries scan from trench center toward the wall, then candidates are prefiltered by line consistency before fitting near a configurable target height (`base_height_pct` plus side-specific offsets). After fitting, selected taper overlay points are filtered by `taper_residual_limit_px` so scratches and top/bottom artifacts do not dominate the displayed wall edge.

## UI

The UI uses a dark metrology dashboard layout:

- Left panel: image list and per-image result summary
- Center: FIB/SEM image viewer with ROI and overlay layers
- Right inspector: measurement settings, overlay controls, candidate summary, and selected result

Overlay layers:

- Raw candidates: low-opacity debug ticks, hidden by default
- Selected edges: brighter markers and measurement line
- Taper selected boundary: highlighted points
- Fit line: separate taper line
- ROI and legend

## CSV Export

CSV export keeps the previous measurement fields and adds debug fields such as raw edge count, coverage (`candidate_coverage` for backward compatibility), selected point count, pair candidate count, per-mode raw edge count, scanline coverage, and threshold.

## Known Limitations

- Raw grayscale jumps are not guaranteed to be physical boundaries.
- Internal layers, shadows, charging artifacts, texture, and noise can all appear as candidates.
- The current goal is transparency: selected candidates and capped raw candidates are preserved and shown instead of hidden behind a binary mask.
- Dense ROIs can still produce many debug ticks; use the raw candidate layer toggle when needed.

## Tests

Run tests from the parent directory:

```powershell
cd C:\Users\admin\Desktop\python
python -m unittest discover -s fib_sem_measurement_tool\tests -v
```
