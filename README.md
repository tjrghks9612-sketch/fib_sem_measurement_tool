# FIB/SEM Measurement Tool

Desktop FIB/SEM metrology tool for measuring CD, THK, single taper, and double taper inside an ROI.

The current measurement pipeline is a raw grayscale edge candidate scanner. It is designed to expose every grayscale jump candidate in the ROI, not to hide candidates behind smoothing, denoising, or confidence heuristics.

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

The measurement path is intentionally simple:

1. Input image
2. Convert to grayscale with `to_gray(image)`
3. Pass raw grayscale directly into measurement functions
4. Scan every row or every column inside the ROI
5. Emit raw edge candidates from direct adjacent grayscale differences

No preprocessing is applied to CD, THK, or taper measurement. The measurement path does not use Gaussian blur, median blur, moving-average smoothing, profile smoothing, morphology, Canny, Sobel, threshold images, or image enhancement.

## Raw Edge Candidate Definition

For each 1D grayscale profile:

```python
data = np.asarray(profile, dtype=np.float32)
diff = data[1:] - data[:-1]
strength = abs(diff)
position = i + 0.5
```

A candidate is emitted when:

```python
abs(diff[i]) >= settings.minimum_grayscale_delta
```

Each candidate stores scan axis, original image scanline index, ROI-local scanline index, subpixel edge position, image coordinate, strength, signed delta, sign, and grayscale values before/after the jump.

## CD and THK

CD and THK are separated into two phases:

1. Generate all raw edge candidates for every ROI row or column.
2. Generate all same-scanline pair candidates for debug/export compatibility.
3. Select boundary pairs from the first threshold-valid edge found in the configured scan direction for each side, with a small continuity refinement for selected overlay/measurement points.

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

Taper measurement first gathers every horizontal raw edge candidate inside the ROI. The left and right boundaries use the same direction-based first-valid selection as CD, followed by a small line-continuity refinement for the fit while preserving all raw candidates in the result and overlay.

## UI

The UI uses a dark metrology dashboard layout:

- Left panel: image list and per-image result summary
- Center: FIB/SEM image viewer with ROI and overlay layers
- Right inspector: measurement settings, overlay controls, candidate summary, selected result, calibration
- Bottom panel: result table for batch review

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
- The current goal is transparency: candidates are preserved and shown instead of filtered away.
- Automatic final pair or boundary selection is intentionally simple and direction based.
- Dense ROIs can produce many debug ticks; use the raw candidate layer toggle when needed.

## Tests

Run tests from the parent directory:

```powershell
cd C:\Users\admin\Desktop\python
python -m unittest discover -s fib_sem_measurement_tool\tests -v
```
