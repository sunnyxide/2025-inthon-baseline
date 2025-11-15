# EC Consistency Loss 및 Hard-Case 커리큘럼 구현 요약

> **작성일**: 2025년 11월 16일  
> **Branch**: feature/model-evaluation-script  
> **Commits**: 637339a, e6735f5

---

## 1. 구현 완료 항목 ✅

### 1-1. EC Consistency Loss
```python
# Location: train.py
def compute_ec_consistency_loss(
    logits, target_output, meta_list, pad_id, lambda_consistency
) -> torch.Tensor | None
```

**기능**:
- 같은 `group_id`를 가진 동치 수식 쌍/트리플의 출력 분포(softmax)를 L2로 일치시킴
- `a+b` vs `b+a`, `(a+b)+c` vs `a+(b+c)` 등 교환/결합법칙 학습 강화
- Law Preservation (20%) / Expression Consistency (30%) 지표 직접 타겟팅

**설정**:
- `lambda_ec = 0.1` (기본 활성화)
- wandb 로깅: `train/loss_ec`

### 1-2. Hard-Case 데이터 강화

#### dataloader.py 개선사항
```python
# LP/Hard_LP 비중 증가: 괄호 우선순위 강화
# EC/Hard_EC 비중 증가: 동치식 쌍/트리플 생성
# 큰수(4~7자리) 곱셈 비율 증가
# 항등원(+0, *1, *0) 패턴 추가
# group_id 메타데이터 추가 (EC consistency용)
```

**목표 카테고리**:
- LP: 19.7% → target >30%
- Hard_EC: 0.5% → target >20%
- OOD (큰수): 23% → target >40%

#### 3) Validation 샘플 개선
```python
# train.py: 쉬운 케이스 대신 hard-case 우선 표시
priority_categories = [
    "parentheses",      # 괄호 (LP)
    "op3_plus",        # 연산자 3개 이상
    "large_number",    # 5+자리 결과
    "mixed",           # 혼합 연산
    "identity",        # 항등원 (+0, *1)
    ...
]
```

### 1-3. 하이퍼파라미터 최적화

```python
# config.py: TrainConfig 기본값 조정
lr = 2e-4                    # 3e-4 → 2e-4 (안정적 수렴)
warmup_steps = 7000          # 8000 → 7000
max_train_steps = 120_000    # None → 120k (긴 학습 허용)
min_lr_threshold = 1e-5      # 1e-6 → 1e-5 (유효 학습 구간 증가)
num_epochs = 15              # 10 → 15 (조금 더 긴 학습)
```

**효과**:
- Plateau 이후에도 유효 학습 지속 (lr > 1e-5)
- 120k step까지 학습 가능 (기존 ~100k에서 멈췄던 것 개선)

### 1-4. 선택적 실험 기능들 (기본값으로 비활성화)

#### A) Digit-level 1D Convolution
```python
# config.py
use_digit_conv = False  # 기본 비활성화

# model.py: TransformerSeq2Seq
if self.use_digit_conv:
    self.digit_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
```

**용도**: 큰수 곱셈의 자리올림 패턴 강화 (필요시 활성화)

#### B) RPN Stack Value Supervision
```python
# config.py
lambda_rpn_value = 0.0  # 기본 비활성화

# model.py
if use_rpn_value_head:
    self.rpn_value_out = nn.Linear(d_model, 1)
```

**용도**: RPN 중간 계산 값을 scratchpad supervision으로 학습 (실험용)

#### C) Plateau-based LR Scheduler
```python
# config.py
use_plateau_schedule = False  # 기본 비활성화

# train.py
if train_config.use_plateau_schedule:
    scheduler = ReduceLROnPlateau(optim, mode='max', ...)
```

**용도**: Validation EM 기준 동적 LR 조정 (cosine 대신)

---

## 2. 규칙 준수 검증 ✅

### 제3조 ②항: 음수 결과 금지
- ✅ **Before**: `_compute_rpn_stack_values()`에서 `res < 0` → `0` 클램핑
- ✅ **After**: 음수 클램핑 제거, 주석으로 "데이터 생성에서 이미 방지됨" 명시
- ✅ **이유**: 모든 생성 함수(`_gen_*`)에서 뺄셈 시 `left >= right` 보장

### 제4조 ①항: 명시적 계산 금지
- ✅ `eval()`은 데이터 생성/검증 시에만 사용 (train/dataloader)
- ✅ `Model.predict()`에는 `eval()`, `int()` 등 계산 함수 없음
- ✅ 모든 계산은 신경망 forward로만 수행

### 제5조 ④항: 사후 보정 금지
- ✅ EC consistency loss는 **train-time auxiliary loss**로만 사용
- ✅ `predict()`에서는 EC loss 전혀 사용 안 함
- ✅ 출력 분포 보정/재랭킹 없음

### 제7조: 입력 전처리 금지
- ✅ `input_text`는 원본 그대로 사용 (문자 단위 인코딩만)
- ✅ RPN은 auxiliary decoder용으로만, 입력 변형 아님
- ✅ 괄호/연산자 순서 변경 없음

---

## 3. 의도하지 않은 변경 체크 ✅

| 항목 | Before | After | 의도 | 상태 |
|------|--------|-------|------|------|
| 기본 모델 구조 | Transformer(6/2) | Transformer(6/2) | 유지 | ✅ |
| nhead | 2 | 2 | Checkpoint 호환 | ✅ |
| RPN auxiliary loss | 0.2 | 0.2 | 유지 | ✅ |
| 데이터 생성 로직 | 40만 base | 40만 base | 유지 | ✅ |
| 증강 로직 | group_id 없음 | group_id 추가 | 의도된 변경 | ✅ |
| 음수 방지 | dataloader | dataloader | 위치 유지 | ✅ |
| Early stopping | Yes | Yes | 유지 | ✅ |

### 불필요한 변경 없음 ✅
- 기존 코드 로직은 모두 유지
- 새 기능은 config 플래그로 제어
- 기본값은 안전하게 설정 (실험 기능 비활성화)

---

## 4. 개선 목표 및 예상 효과

### 벤치마크 약점 → 개선 방향

| 카테고리 | 현재 | 목표 | 개선 방법 |
|----------|------|------|-----------|
| Regular LP | 19.7% | >30% | 괄호/우선순위 데이터 증가 |
| Hard_EC | 0.5% | >20% | EC consistency loss + 동치식 쌍 |
| Hard_LP | 0.0% | >10% | 깊은 괄호 패턴 추가 |
| OOD | 23% | >40% | 4~7자리 큰수 비율 증가 |
| Hard_RC | 25.5% | >35% | 나눗셈 포함 복합식 증가 |

### Public Benchmark 예상 개선
```
기존 (단조 RPN):
  Overall: 0.6355
  Calculation: 35% → target >45%
  Law: 20% → target >35%
  Expression: 30% → target >50%
  Relational: 15% → target >20%

개선 요소:
  ✅ EC consistency loss (교환/결합/분배 법칙)
  ✅ Hard-case 데이터 (LP/EC/큰수)
  ✅ 긴 학습 (120k steps, lr 안정화)
  ✅ 향상된 validation 모니터링
```

---

## 5. Git Commit 이력

```bash
637339a - feat: Add EC consistency loss and hard-case curriculum
          (config, dataloader, model, docs 전체 업데이트)

e6735f5 - fix: Add EC consistency loss integration and remove negative clamping
          (train.py에 compute_ec_consistency_loss 추가 및 음수 클램핑 제거)
```

---

## 6. 학습 시작 명령어

```bash
cd /Users/sunny/datathon/2025-inthon-baseline
source ../venv/bin/activate
python train.py
```

### 예상 동작
1. Base 40만개 데이터 생성
2. 증강 60만개로 확장 (group_id 포함)
3. Checkpoint 로드 (best_model.pt)
4. 학습 시작:
   - train/loss, train/loss_rpn, train/loss_ec, train/lr
   - valid/EM, valid/TES
5. Hard-case 샘플 우선 표시
6. Best EM 도달 시 자동 저장

### 모니터링 포인트
- `train/loss_ec`: 0.01~0.1 범위면 정상
- Hard_EC/Hard_LP 샘플 정확도 추적
- lr이 1e-5 이하로 떨어지기 전 수렴 확인

---

## 7. 최종 체크리스트 ✅

- [x] EC consistency loss 구현 및 통합
- [x] 음수 클램핑 제거 (규칙 준수)
- [x] Hard-case 데이터 강화
- [x] group_id 메타데이터 추가
- [x] Validation 샘플링 개선
- [x] 하이퍼파라미터 최적화
- [x] 선택적 기능들 비활성화
- [x] 문법 오류 없음
- [x] Git commit 완료
- [x] 규칙 준수 검증

**상태**: ✅ 학습 준비 완료

---

**작성자**: AI Assistant  
**검토 완료**: 2025년 11월 16일

