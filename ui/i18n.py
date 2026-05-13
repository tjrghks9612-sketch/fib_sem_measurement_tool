from __future__ import annotations

from typing import Dict


LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "vi": "Tiếng Việt",
}
LANGUAGE_BY_LABEL = {label: code for code, label in LANGUAGES.items()}


MEASUREMENT_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {
        "taper_single": "단일 테이퍼",
        "taper_double": "양측 테이퍼",
        "distance_horizontal": "가로 CD",
        "distance_vertical": "세로 THK",
        "distance_both": "가로 + 세로",
    },
    "en": {
        "taper_single": "Single taper",
        "taper_double": "Double taper",
        "distance_horizontal": "Horizontal CD",
        "distance_vertical": "Vertical THK",
        "distance_both": "Horizontal + Vertical",
    },
    "vi": {
        "taper_single": "Taper một bên",
        "taper_double": "Taper hai bên",
        "distance_horizontal": "CD ngang",
        "distance_vertical": "THK dọc",
        "distance_both": "Ngang + Dọc",
    },
}

DISTANCE_METHOD_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"mean": "평균", "max": "최대", "min": "최소"},
    "en": {"mean": "Mean", "max": "Max", "min": "Min"},
    "vi": {"mean": "Trung bình", "max": "Lớn nhất", "min": "Nhỏ nhất"},
}

EDGE_SCAN_MODE_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"auto": "자동", "outside_to_center": "바깥쪽 -> 중심", "center_to_outside": "중심 -> 바깥쪽"},
    "en": {"auto": "Auto", "outside_to_center": "Outside -> center", "center_to_outside": "Center -> outside"},
    "vi": {"auto": "Tự động", "outside_to_center": "Ngoài -> tâm", "center_to_outside": "Tâm -> ngoài"},
}

PROFILE_MODE_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"both": "둘 다", "horizontal": "가로", "vertical": "세로"},
    "en": {"both": "Both", "horizontal": "Horizontal", "vertical": "Vertical"},
    "vi": {"both": "Cả hai", "horizontal": "Ngang", "vertical": "Dọc"},
}

TAPER_SIDE_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"left": "좌측", "right": "우측"},
    "en": {"left": "Left", "right": "Right"},
    "vi": {"left": "Trái", "right": "Phải"},
}

STATUS_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"OK": "정상", "Check": "확인", "Review Needed": "검토 필요", "Fail": "실패", "Not measured": "측정 전"},
    "en": {"OK": "OK", "Check": "Check", "Review Needed": "Review needed", "Fail": "Fail", "Not measured": "Not measured"},
    "vi": {"OK": "OK", "Check": "Kiểm tra", "Review Needed": "Cần xem lại", "Fail": "Lỗi", "Not measured": "Chưa đo"},
}

SETTINGS_SOURCE_LABELS: Dict[str, Dict[str, str]] = {
    "ko": {"global_default": "기본 설정", "image_specific": "이미지별 설정"},
    "en": {"global_default": "Default settings", "image_specific": "Image settings"},
    "vi": {"global_default": "Cài đặt mặc định", "image_specific": "Cài đặt ảnh"},
}

STRINGS: Dict[str, Dict[str, str]] = {
    "ko": {
        "app_title": "FIB-SEM 측정 도구",
        "toolbar_title": "FIB/SEM 측정 도구",
        "current_file_none": "선택된 파일 없음",
        "initial_status": "이미지를 불러오고 ROI를 드래그한 뒤 측정하세요.",
        "load_images": "이미지 불러오기",
        "load_folder": "폴더 불러오기",
        "reset_images": "이미지 초기화",
        "previous": "이전",
        "next": "다음",
        "measure_current": "현재 측정",
        "measure_all": "전체 측정",
        "save_csv": "CSV 저장",
        "language": "언어",
        "option_header": "검사 설정",
        "section_measurement": "측정 설정",
        "measurement_mode": "측정 모드",
        "taper_side": "테이퍼 측",
        "representative_value": "대표값",
        "edge_scan_start": "탐색 시작",
        "normalize_signal": "신호 정규화",
        "denoise_signal": "신호 스무딩",
        "minimum_delta": "최소 변화량",
        "boundary_angle_filter": "CD/THK 각도 필터",
        "max_boundary_angle": "허용 경계 각도",
        "taper_height": "테이퍼 높이",
        "taper": "테이퍼",
        "section_overlay": "오버레이",
        "selected_edges": "선택 경계",
        "fit_line": "피팅 선",
        "labels": "라벨",
        "section_candidate_summary": "후보 요약",
        "raw_edge_count": "원시 경계 수",
        "confidence": "신뢰도",
        "selected_points": "선택 포인트",
        "pair_candidates": "쌍 후보",
        "threshold": "임계값",
        "section_selected_result": "선택 결과",
        "section_calibration": "캘리브레이션",
        "detect_scale_bar": "스케일바 검출",
        "detected_px": "검출 px",
        "actual_length": "실제 길이",
        "unit": "단위",
        "apply_calibration": "캘리브레이션 적용",
        "before_measurement": "측정 전",
        "no_selected_result": "선택 결과 없음",
        "average_taper": "평균 테이퍼",
        "left_taper": "좌측 테이퍼",
        "right_taper": "우측 테이퍼",
        "ellipse_cd_horizontal": "CD 타원 가로",
        "ellipse_cd_vertical": "CD 타원 세로",
        "gray_suffix": "그레이",
        "viewer_title": "이미지 뷰어",
        "load_image_prompt": "이미지를 불러오세요.",
        "fit": "맞춤",
        "overlay": "오버레이",
        "profile_title": "그레이스케일 그래프",
        "profile_hover_prompt": "이미지에 마우스를 올리면 그레이스케일 그래프를 표시합니다",
        "profile_no_image": "이미지 없음",
        "profile_load_image": "이미지를 불러오세요",
        "profile_hover_main": "메인 이미지에 마우스를 올리세요",
        "profile_original": "원본",
        "profile_graph_basis": "그래프",
        "profile_roi": "ROI",
        "profile_full": "전체",
        "profile_normalized": "정규화",
        "profile_smoothed": "스무딩",
        "thumb_header": "이미지 / 결과",
        "all_types": "전체 유형",
        "all_statuses": "전체 상태",
        "images_unit": "개 이미지",
        "select_all": "전체 선택",
        "clear": "해제",
        "delete_selected_images": "선택 이미지 삭제",
        "measure_selected": "선택 측정",
        "selection": "선택",
        "no_preview": "미리보기 없음",
        "roi_exists": "ROI 있음",
        "roi_missing": "ROI 없음",
        "calibration": "캘리브레이션",
        "not_calibrated": "미보정",
        "calibrated": "보정됨",
        "loaded_images": "이미지 {count}개를 불러왔습니다.",
        "selected_images_deleted": "선택 이미지 {count}개를 삭제했습니다.",
        "no_selected_images": "선택된 이미지가 없습니다.",
        "load_failures": " 실패 {count}개.",
        "load_image_failed": "이미지 불러오기 실패",
        "roi_too_small": "ROI가 너무 작습니다.",
        "roi_applied": "현재 이미지에 ROI를 적용했습니다.",
        "scale_bar_detected": "스케일바 후보 검출: {pixel_length:.2f} px",
        "scale_bar": "스케일바",
        "scale_bar_detection": "스케일바 검출",
        "scale_bar_failed": "스케일바 검출 실패",
        "calibration_failed": "캘리브레이션 실패",
        "calibration_failed_message": "스케일바 검출 또는 실제 길이 입력을 확인하세요.",
        "default_calibration_applied": "기본 캘리브레이션 적용 ({scale:.6g} {unit}/px)",
        "image_calibration_applied": "{file_name}: 캘리브레이션 적용 ({scale:.6g} {unit}/px)",
        "measuring": "측정 중 {index}/{total}: {file_name}",
        "measurement_complete": "이미지 {count}개 측정 완료.",
        "roi_applied_count": " / ROI 적용 {count}개",
        "failures_count": " / 실패 {count}개",
        "csv_saved": "CSV 저장 완료: {path}",
        "csv_saved_title": "CSV 저장",
        "csv_saved_message": "결과 CSV를 저장했습니다.",
        "option_changed": "옵션 변경: {file_name} / {status} {confidence:.0f}%",
        "image_file_dialog_title": "FIB-SEM 이미지 선택",
        "image_files": "이미지 파일",
        "all_files": "모든 파일",
        "folder_dialog_title": "이미지 폴더 선택",
    },
    "en": {
        "app_title": "FIB-SEM Measurement Tool",
        "toolbar_title": "FIB/SEM Measurement Tool",
        "current_file_none": "No file selected",
        "initial_status": "Load images, drag an ROI, then measure.",
        "load_images": "Load Images",
        "load_folder": "Load Folder",
        "reset_images": "Reset Images",
        "previous": "Previous",
        "next": "Next",
        "measure_current": "Measure Current",
        "measure_all": "Measure All",
        "save_csv": "Save CSV",
        "language": "Language",
        "option_header": "Inspection Settings",
        "section_measurement": "Measurement",
        "measurement_mode": "Mode",
        "taper_side": "Taper Side",
        "representative_value": "Representative",
        "edge_scan_start": "Scan Start",
        "normalize_signal": "Normalize Signal",
        "denoise_signal": "Smooth Signal",
        "minimum_delta": "Minimum Delta",
        "boundary_angle_filter": "CD/THK Angle Filter",
        "max_boundary_angle": "Max Boundary Angle",
        "taper_height": "Taper Height",
        "taper": "taper",
        "section_overlay": "Overlay",
        "selected_edges": "Selected Edges",
        "fit_line": "Fit Line",
        "labels": "Labels",
        "section_candidate_summary": "Candidate Summary",
        "raw_edge_count": "Raw Edges",
        "confidence": "Confidence",
        "selected_points": "Selected Points",
        "pair_candidates": "Pair Candidates",
        "threshold": "Threshold",
        "section_selected_result": "Selected Result",
        "section_calibration": "Calibration",
        "detect_scale_bar": "Detect Scale Bar",
        "detected_px": "Detected px",
        "actual_length": "Actual Length",
        "unit": "Unit",
        "apply_calibration": "Apply Calibration",
        "before_measurement": "Before measurement",
        "no_selected_result": "No selected result",
        "average_taper": "Average taper",
        "left_taper": "Left taper",
        "right_taper": "Right taper",
        "ellipse_cd_horizontal": "Ellipse CD H",
        "ellipse_cd_vertical": "Ellipse CD V",
        "gray_suffix": "gray",
        "viewer_title": "Image Viewer",
        "load_image_prompt": "Load an image.",
        "fit": "Fit",
        "overlay": "Overlay",
        "profile_title": "Grayscale Profile",
        "profile_hover_prompt": "Hover over the image to show the grayscale profile",
        "profile_no_image": "No image",
        "profile_load_image": "Load an image",
        "profile_hover_main": "Hover over the main image",
        "profile_original": "Raw",
        "profile_graph_basis": "Graph",
        "profile_roi": "ROI",
        "profile_full": "Full",
        "profile_normalized": "Normalized",
        "profile_smoothed": "Smoothed",
        "thumb_header": "Images / Results",
        "all_types": "All types",
        "all_statuses": "All statuses",
        "images_unit": "images",
        "select_all": "Select All",
        "clear": "Clear",
        "delete_selected_images": "Delete Selected",
        "measure_selected": "Measure Selected",
        "selection": "Selected",
        "no_preview": "No preview",
        "roi_exists": "ROI set",
        "roi_missing": "No ROI",
        "calibration": "Calibration",
        "not_calibrated": "Not calibrated",
        "calibrated": "Calibrated",
        "loaded_images": "Loaded {count} images.",
        "selected_images_deleted": "Deleted {count} selected images.",
        "no_selected_images": "No selected images.",
        "load_failures": " Failed {count}.",
        "load_image_failed": "Failed to load images",
        "roi_too_small": "ROI is too small.",
        "roi_applied": "Applied ROI to the current image.",
        "scale_bar_detected": "Scale bar candidate: {pixel_length:.2f} px",
        "scale_bar": "Scale bar",
        "scale_bar_detection": "Scale bar detection",
        "scale_bar_failed": "Scale bar detection failed",
        "calibration_failed": "Calibration failed",
        "calibration_failed_message": "Check scale bar detection or the actual length input.",
        "default_calibration_applied": "Applied default calibration ({scale:.6g} {unit}/px)",
        "image_calibration_applied": "{file_name}: applied calibration ({scale:.6g} {unit}/px)",
        "measuring": "Measuring {index}/{total}: {file_name}",
        "measurement_complete": "Measured {count} images.",
        "roi_applied_count": " / ROI applied to {count}",
        "failures_count": " / failures {count}",
        "csv_saved": "CSV saved: {path}",
        "csv_saved_title": "Save CSV",
        "csv_saved_message": "Saved result CSV.",
        "option_changed": "Option changed: {file_name} / {status} {confidence:.0f}%",
        "image_file_dialog_title": "Select FIB-SEM Images",
        "image_files": "Image files",
        "all_files": "All files",
        "folder_dialog_title": "Select Image Folder",
    },
    "vi": {
        "app_title": "Công cụ đo FIB-SEM",
        "toolbar_title": "Công cụ đo FIB/SEM",
        "current_file_none": "Chưa chọn tệp",
        "initial_status": "Tải ảnh, kéo ROI rồi đo.",
        "load_images": "Tải ảnh",
        "load_folder": "Tải thư mục",
        "reset_images": "Đặt lại ảnh",
        "previous": "Trước",
        "next": "Sau",
        "measure_current": "Đo ảnh hiện tại",
        "measure_all": "Đo tất cả",
        "save_csv": "Lưu CSV",
        "language": "Ngôn ngữ",
        "option_header": "Cài đặt kiểm tra",
        "section_measurement": "Đo lường",
        "measurement_mode": "Chế độ",
        "taper_side": "Phía taper",
        "representative_value": "Giá trị đại diện",
        "edge_scan_start": "Bắt đầu quét",
        "normalize_signal": "Chuẩn hóa tín hiệu",
        "denoise_signal": "Làm mượt tín hiệu",
        "minimum_delta": "Delta tối thiểu",
        "boundary_angle_filter": "Lọc góc CD/THK",
        "max_boundary_angle": "Góc biên tối đa",
        "taper_height": "Chiều cao taper",
        "taper": "taper",
        "section_overlay": "Lớp phủ",
        "selected_edges": "Biên đã chọn",
        "fit_line": "Đường fit",
        "labels": "Nhãn",
        "section_candidate_summary": "Tóm tắt ứng viên",
        "raw_edge_count": "Số biên thô",
        "confidence": "Độ tin cậy",
        "selected_points": "Điểm đã chọn",
        "pair_candidates": "Cặp ứng viên",
        "threshold": "Ngưỡng",
        "section_selected_result": "Kết quả đã chọn",
        "section_calibration": "Hiệu chuẩn",
        "detect_scale_bar": "Dò scale bar",
        "detected_px": "px dò được",
        "actual_length": "Chiều dài thật",
        "unit": "Đơn vị",
        "apply_calibration": "Áp dụng hiệu chuẩn",
        "before_measurement": "Chưa đo",
        "no_selected_result": "Không có kết quả",
        "average_taper": "Taper trung bình",
        "left_taper": "Taper trái",
        "right_taper": "Taper phải",
        "ellipse_cd_horizontal": "CD ellipse ngang",
        "ellipse_cd_vertical": "CD ellipse dọc",
        "gray_suffix": "gray",
        "viewer_title": "Trình xem ảnh",
        "load_image_prompt": "Tải một ảnh.",
        "fit": "Vừa khung",
        "overlay": "Lớp phủ",
        "profile_title": "Biểu đồ grayscale",
        "profile_hover_prompt": "Di chuột lên ảnh để xem biểu đồ grayscale",
        "profile_no_image": "Không có ảnh",
        "profile_load_image": "Tải một ảnh",
        "profile_hover_main": "Di chuột lên ảnh chính",
        "profile_original": "Gốc",
        "profile_graph_basis": "Biểu đồ",
        "profile_roi": "ROI",
        "profile_full": "Toàn ảnh",
        "profile_normalized": "Đã chuẩn hóa",
        "profile_smoothed": "Đã làm mượt",
        "thumb_header": "Ảnh / Kết quả",
        "all_types": "Tất cả loại",
        "all_statuses": "Tất cả trạng thái",
        "images_unit": "ảnh",
        "select_all": "Chọn tất cả",
        "clear": "Bỏ chọn",
        "delete_selected_images": "Xóa ảnh đã chọn",
        "measure_selected": "Đo ảnh đã chọn",
        "selection": "Đã chọn",
        "no_preview": "Không có xem trước",
        "roi_exists": "Có ROI",
        "roi_missing": "Không có ROI",
        "calibration": "Hiệu chuẩn",
        "not_calibrated": "Chưa hiệu chuẩn",
        "calibrated": "Đã hiệu chuẩn",
        "loaded_images": "Đã tải {count} ảnh.",
        "selected_images_deleted": "Đã xóa {count} ảnh đã chọn.",
        "no_selected_images": "Không có ảnh nào được chọn.",
        "load_failures": " Lỗi {count}.",
        "load_image_failed": "Tải ảnh thất bại",
        "roi_too_small": "ROI quá nhỏ.",
        "roi_applied": "Đã áp dụng ROI cho ảnh hiện tại.",
        "scale_bar_detected": "Ứng viên scale bar: {pixel_length:.2f} px",
        "scale_bar": "Scale bar",
        "scale_bar_detection": "Dò scale bar",
        "scale_bar_failed": "Dò scale bar thất bại",
        "calibration_failed": "Hiệu chuẩn thất bại",
        "calibration_failed_message": "Kiểm tra scale bar hoặc chiều dài thật.",
        "default_calibration_applied": "Đã áp dụng hiệu chuẩn mặc định ({scale:.6g} {unit}/px)",
        "image_calibration_applied": "{file_name}: đã áp dụng hiệu chuẩn ({scale:.6g} {unit}/px)",
        "measuring": "Đang đo {index}/{total}: {file_name}",
        "measurement_complete": "Đã đo {count} ảnh.",
        "roi_applied_count": " / ROI áp dụng cho {count}",
        "failures_count": " / lỗi {count}",
        "csv_saved": "Đã lưu CSV: {path}",
        "csv_saved_title": "Lưu CSV",
        "csv_saved_message": "Đã lưu CSV kết quả.",
        "option_changed": "Đã đổi tùy chọn: {file_name} / {status} {confidence:.0f}%",
        "image_file_dialog_title": "Chọn ảnh FIB-SEM",
        "image_files": "Tệp ảnh",
        "all_files": "Tất cả tệp",
        "folder_dialog_title": "Chọn thư mục ảnh",
    },
}


def normalize_language(language: str) -> str:
    return language if language in LANGUAGES else "ko"


def language_label(language: str) -> str:
    return LANGUAGES.get(normalize_language(language), LANGUAGES["ko"])


def language_code(label: str) -> str:
    return LANGUAGE_BY_LABEL.get(label, "ko")


def t(language: str, key: str) -> str:
    language = normalize_language(language)
    return STRINGS.get(language, {}).get(key, STRINGS["ko"].get(key, key))


def _label(mapping: Dict[str, Dict[str, str]], language: str, key: str) -> str:
    language = normalize_language(language)
    return mapping.get(language, {}).get(key, mapping["ko"].get(key, key))


def _key_from_label(mapping: Dict[str, Dict[str, str]], label: str, default: str) -> str:
    for language_map in mapping.values():
        for key, value in language_map.items():
            if value == label:
                return key
    return default


def measurement_label(language: str, key: str) -> str:
    return _label(MEASUREMENT_LABELS, language, key)


def measurement_key(label: str, default: str = "distance_both") -> str:
    return _key_from_label(MEASUREMENT_LABELS, label, default)


def distance_method_label(language: str, key: str) -> str:
    return _label(DISTANCE_METHOD_LABELS, language, key)


def distance_method_key(label: str, default: str = "mean") -> str:
    return _key_from_label(DISTANCE_METHOD_LABELS, label, default)


def edge_scan_mode_label(language: str, key: str) -> str:
    return _label(EDGE_SCAN_MODE_LABELS, language, key)


def edge_scan_mode_key(label: str, default: str = "auto") -> str:
    return _key_from_label(EDGE_SCAN_MODE_LABELS, label, default)


def profile_mode_label(language: str, key: str) -> str:
    return _label(PROFILE_MODE_LABELS, language, key)


def profile_mode_key(label: str, default: str = "both") -> str:
    return _key_from_label(PROFILE_MODE_LABELS, label, default)


def taper_side_label(language: str, key: str) -> str:
    return _label(TAPER_SIDE_LABELS, language, key)


def status_label(language: str, status: str) -> str:
    return _label(STATUS_LABELS, language, status)


def settings_source_label(language: str, source: str) -> str:
    return _label(SETTINGS_SOURCE_LABELS, language, source)
