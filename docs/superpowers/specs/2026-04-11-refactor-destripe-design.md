# Refactor Spec — destripe (2026-04-11)

## Scope

Diagnosed the full `destripe` codebase:
- `src/destripe/ops.py` (numpy public API)
- `src/destripe/core.py` (PDHG torch solver + tiling)
- `src/destripe/__init__.py`
- `tests/test_core.py`

Run via `/refactor` with both Feynman and MECE lenses.

## Context summary

- `CLAUDE.md` — two-layer design (numpy boundary over torch solver); code conventions (type hints, private `_` prefix, docstrings on public API only).
- `README.md` — user-facing features and parameter guide.
- `git log` — most recent work (`b8dd14d refactor: fix misleading gradient var names and clean up core/ops`) touched the same solver internals this spec now revisits.
- `tests/test_core.py` — 36 tests covering adjoint consistency, `process`/`process_tiled` shapes/dtypes, `destripe()` dtype round-trip, and all validation error paths.

## Selected findings

### F-01: `ops.py`의 파라미터 검증은 `core`의 검증과 중복
- **Location**: `src/destripe/ops.py:49-56`
- **Category**: duplication (MECE)
- **Observation**: `iterations`, `tol`, `tiles`, `overlap`에 대한 4개의 동일한 `ValueError` raise가 `ops.destripe()`와 `core._validate_solver_params` / `_validate_tiling_params` 두 군데에 존재. 에러 메시지까지 바이트 단위로 동일.
- **Reconstruction attempt**: "공용 함수가 빠른 실패를 위해 입력을 먼저 검증한다." — 하지만 core의 `process` / `process_tiled`가 이미 동일 검증을 수행하며, core는 공개 클래스이므로 해당 검증이 **권위 있는** 레이어. ops의 복제는 core를 신뢰하지 않는다는 의미인데 그렇게 믿지 않을 이유가 없음.
- **Failure point**: "왜 두 곳에 있어야 하는가"에 답할 수 없음. 한 곳이면 충분.
- **Suggested direction**: `ops.destripe()`는 numpy 고유 검증(dtype/shape/finite)만 수행하고 solver 파라미터 검증은 core에 위임. 단, F-02와 함께 처리해야 함 — 현재 `_run`이 `tiles=1`일 때 `process`(tiles 검증 없음)를 호출하므로, F-02를 먼저 적용해 항상 `process_tiled`로 통일한 뒤 ops 검증 제거.
- **Axes**: Impact: med, Confidence: high, Effort: S
- **Verification**: 워크트리 실험 — `ops.py`의 검증 8줄 제거 + `_run`을 항상 `process_tiled` 호출로 단순화 → `pytest tests/` 36 passed. ops 검증이 순수 중복임이 실행으로 확인됨.
- **Conflicts**: F-02와 함께 처리해야 함 (의존 관계, 상충 아님).

### F-02: `_run`의 `tiles>1` 분기는 사역(dead)
- **Location**: `src/destripe/ops.py:115-132`
- **Category**: dead-code (Feynman)
- **Observation**: `_run`은 `tiles>1`이면 `process_tiled`, 아니면 `process`로 분기. 그런데 `process_tiled`는 `core.py:128-135`에서 `tiles<=1`일 때 `process`로 직접 fallback한다.
- **Reconstruction attempt**: "타일이 없을 때의 빠른 경로를 제공한다." — 하지만 `process_tiled`의 fallback도 정확히 동일하게 빠른 경로로 위임한다. `_run`의 분기가 없어도 동작·성능 동일.
- **Failure point**: 분기가 제거될 수 있는데 남아 있음. 이유를 재구성할 수 없음.
- **Suggested direction**: `_run`을 삭제하고 `destripe()`에서 `remover.process_tiled(...)`를 직접 호출. 또는 `_run`은 남기되 내부 분기만 제거.
- **Axes**: Impact: low, Confidence: high, Effort: S
- **Verification**: F-01과 함께 실험에서 분기 제거 후 36 passed. Confirmed.

### F-03: `correction` 버퍼의 이름이 사용 의미와 불일치
- **Location**: `src/destripe/core.py:234` 선언, `_solve` 본문 전반 (lines 260-292)
- **Category**: naming (MECE) / complexity (Feynman)
- **Observation**: 동일 버퍼가 (a) `u+Σs=data` 제약의 공유 잔차(260-266), (b) `proj` 클램프 잔차 분배(271-275), (c) `forward_diff`의 출력(281-284), (d) TV 듀얼의 스케일링 승수(289-292) 네 가지 무관한 값으로 순차 재사용됨. 같은 이름이 네 번 재정의.
- **Reconstruction attempt**: "할당을 피하기 위한 pre-allocated 스크래치 버퍼." — 버퍼 재사용 자체는 정당하지만 이름 `correction`은 (a) 문맥에서만 말이 되고 (c)(d)에서는 misleading. 리뷰어가 289-292를 읽을 때 "왜 TV 승수가 correction에 저장되지?"라고 멈추게 됨.
- **Failure point**: 이름이 목적을 암시하는데 목적이 네 번 바뀜. 최근 커밋 `b8dd14d`가 "fix misleading gradient var names"였던 것을 보면 저자도 이 영역을 정리했었음.
- **Suggested direction**: 중립적인 이름 `scratch`로 바꾸거나, 사용처별로 분리된 버퍼 2~3개 도입(할당 비용 미미). 둘 다 코드 의도를 드러냄. 기존의 `directional_diff`, `grad_norm` 등 용도별 버퍼와 일관성을 맞추는 것이 권장됨.
- **Axes**: Impact: med, Confidence: high, Effort: S
- **Verification**: not falsifiable by execution

## Refactoring constraints

- 기존 `tests/test_core.py`의 36개 테스트는 변경 없이 모두 통과해야 한다 (에러 메시지 매칭 포함: `match="iterations"`, `match="tol"`, `match="tiles"`, `match="overlap"`, `match="shape"`, `match="C in"`, `match="NaN or Inf"`).
- 공개 API 유지: `destripe()` 시그니처와 `UniversalStripeRemover` 공개 메서드 (`process`, `process_tiled`) 시그니처·동작 동일.
- `destripe()`의 수치 결과가 동일해야 한다 (PDHG 반복은 결정적이며 `test_reproducible_output_for_fixed_input`가 이를 보장한다). F-01/F-02의 경로 변경이 `tiles=1`에서 결과를 바꾸면 안 된다 — `process_tiled(tiles=1)`의 fallback이 `process`를 그대로 위임하므로 수치 동치.
- `CLAUDE.md`의 코드 컨벤션 준수: 타입 힌트 (`X | Y`), 공개 API에만 docstring, 사설 메서드 `_` 접두, 경계에서 numpy / 내부에서 torch.

## Success criteria

리팩토링 후 동일한 Feynman / MECE 재구성이 다음 지점에서 더 이상 실패하지 않아야 한다:

1. **F-01**: "왜 `ops.py`와 `core.py` 양쪽에서 동일한 파라미터 검증을 하는가?"라는 질문이 생기지 않는다 — 검증이 단일 레이어(core)에만 존재.
2. **F-02**: `destripe()`에서 solver까지의 경로에서 "왜 여기서 `tiles`를 분기하는가, `process_tiled`가 이미 그걸 처리하는데?"라는 질문이 생기지 않는다 — 중복 분기 제거.
3. **F-03**: `_solve` 본문을 처음 읽는 리뷰어가 스크래치 버퍼의 이름만 보고 "이게 왜 여기서 이 값을 담지?"에서 멈추지 않는다 — 이름이 실제 용도(중립적이거나 용도별)를 반영.
4. **회귀 없음**: `pytest tests/ -v` 36 passed. 이미 F-01/F-02는 워크트리 실험으로 확인됨.
