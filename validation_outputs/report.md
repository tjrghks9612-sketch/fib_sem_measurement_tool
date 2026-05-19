# Sample Measurement Validation

## 가로 CD 1.png
- expected_mode: 가로 CD
- measured_values: horizontal_cd_px=159.4
- confidence: 98.86
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/가로 CD 1_overlay.png
- target_overlay_path: -
- validation: pass
- failure_reason: -

## 가로 CD 2.png
- expected_mode: 가로 CD
- measured_values: horizontal_cd_px=160.7
- confidence: 76.86
- status: Check
- warning_message: -
- overlay_path: validation_outputs/overlays/가로 CD 2_overlay.png
- target_overlay_path: -
- validation: pass
- failure_reason: -

## 세로 THK 1.png
- expected_mode: 세로 THK
- measured_values: vertical_thk_px=234.2
- confidence: 99.14
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/세로 THK 1_overlay.png
- target_overlay_path: -
- validation: pass
- failure_reason: -

## 세로 THK 2.png
- expected_mode: 세로 THK
- measured_values: vertical_thk_px=211.6
- confidence: 99.64
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/세로 THK 2_overlay.png
- target_overlay_path: -
- validation: pass
- failure_reason: -

## 가로+세로 1.png
- expected_mode: 가로+세로
- measured_values: horizontal_cd_px=418.3, vertical_thk_px=485.6
- confidence: 42.92
- status: Review Needed
- warning_message: -
- overlay_path: validation_outputs/overlays/가로+세로 1_overlay.png
- target_overlay_path: -
- validation: partial
- failure_reason: low combined confidence

## 가로+세로 2.png
- expected_mode: 가로+세로
- measured_values: horizontal_cd_px=466.9, vertical_thk_px=548.2
- confidence: 43.67
- status: Review Needed
- warning_message: -
- overlay_path: validation_outputs/overlays/가로+세로 2_overlay.png
- target_overlay_path: -
- validation: partial
- failure_reason: low combined confidence

## Taper 1.png
- expected_mode: Taper
- measured_values: left_taper_deg=83.26, left_taper_points=539, right_taper_deg=83.1, right_taper_points=502
- confidence: 69.74
- status: Check
- warning_message: right dark trench boundary unstable
- overlay_path: validation_outputs/overlays/Taper 1_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Taper 1_target.png
- validation: pass
- failure_reason: -

## Taper 2.png
- expected_mode: Taper
- measured_values: left_taper_deg=73.06, left_taper_points=374, right_taper_deg=72.89, right_taper_points=529
- confidence: 75.27
- status: Check
- warning_message: right dark trench boundary unstable
- overlay_path: validation_outputs/overlays/Taper 2_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Taper 2_target.png
- validation: pass
- failure_reason: -

## Hole CD 1.png
- expected_mode: Hole CD
- measured_values: hole_h_px=300.6, hole_v_px=322.3, hole_target=inner, hole_coverage=1
- confidence: 86.4
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/Hole CD 1_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Hole CD 1_target.png
- validation: pass
- failure_reason: -

## Hole CD 2.png
- expected_mode: Hole CD
- measured_values: hole_h_px=343.2, hole_v_px=336.7, hole_target=inner, hole_coverage=1
- confidence: 85.77
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/Hole CD 2_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Hole CD 2_target.png
- validation: pass
- failure_reason: -

## Crater 1.png
- expected_mode: Crater
- measured_values: crater_cd_px=664, crater_thk_px=154.9, crater_left_taper_deg=40.57, crater_right_taper_deg=40.67
- confidence: 83.83
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/Crater 1_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Crater 1_target.png
- validation: pass
- failure_reason: -

## Crater 2.png
- expected_mode: Crater
- measured_values: crater_cd_px=844, crater_thk_px=152.8, crater_left_taper_deg=31.51, crater_right_taper_deg=31.85
- confidence: 92.28
- status: OK
- warning_message: -
- overlay_path: validation_outputs/overlays/Crater 2_overlay.png
- target_overlay_path: validation_outputs/target_overlays/Crater 2_target.png
- validation: pass
- failure_reason: -
