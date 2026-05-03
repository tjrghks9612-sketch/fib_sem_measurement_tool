# Senior Code Review (2026-05-03)

## Overall
- 구조 분리는 전반적으로 깔끔합니다 (`core`, `ui`, `models`, `export`).
- dataclass 기반 설정 모델링과 설정 merge 흐름이 명확합니다.
- 다만 에러 처리/관측성(observability)/타입 안정성 측면에서 개선 여지가 있습니다.

## High Priority
1. **광범위한 예외 삼키기 (`except Exception`) 최소화 필요**
   - 현재 측정 파이프라인에서 모든 예외를 하나로 처리해 원인 분석이 어렵습니다.
   - 최소한 입력 검증 예외, 수치 연산 예외, ROI 예외를 구분해 사용자 메시지와 로그를 분리하세요.

2. **실패 원인 로깅 경로 부재**
   - UI에는 친화적 메시지를 보여주되, 내부적으로는 traceback/컨텍스트(이미지명, ROI, 측정타입)를 남겨야 합니다.
   - 운영 중 재현 불가 이슈를 줄이려면 구조화 로그(JSON 또는 key=value)가 필요합니다.

## Medium Priority
1. **설정 병합 시 스키마 유효성 검증 권장**
   - `merge_settings`가 dict 기반으로 모든 키를 신뢰합니다.
   - 버전 업 시 누락/오타 키가 조용히 반영될 수 있으므로, 허용 키 화이트리스트 검증을 추가하세요.

2. **`settings_source` 의미론 정리**
   - `resolve_effective_settings`에서 source 복원 로직이 동작하지만, 복사/전파 시점 의미가 혼재될 가능성이 있습니다.
   - "원본 출처"와 "최종 적용 출처"를 별도 필드로 분리하면 디버깅이 쉬워집니다.

3. **측정 status 체계 표준화 권장**
   - `OK`, `Check`, `Review Needed`, `Fail` 문자열 비교는 오타에 취약합니다.
   - Enum 또는 상수 객체로 통일하고 UI 표시 문자열은 별도 매핑하세요.

## Low Priority
1. **노이즈 프리셋 파라미터 하드코딩 테이블화**
   - 조건문 기반 세팅 대신 프리셋 테이블(dict)로 두면 변경 이력 관리가 쉬워집니다.

2. **README 실행 경로 예시 단순화**
   - 패키지 루트 기준 실행 명령을 하나로 통일하면 온보딩 실수를 줄일 수 있습니다.

## Suggested Next Steps
1. `measurement_runner` 예외 분류 + 내부 로깅 추가.
2. status 문자열 Enum화.
3. `merge_settings` 키 검증 레이어 추가.
4. 회귀 테스트: ROI 없음/작음, 캘리브레이션 미적용, 측정타입별 Fail/Check 경계 케이스.
