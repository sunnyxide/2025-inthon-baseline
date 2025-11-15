# 코드 변경 사항 검토 체크리스트

> **작성일**: 2025년 11월 16일  
> **Branch**: feature/ec-consistency-hardcases  
> **Status**: ✅ 검토 완료

---

## 1. 주요 변경 사항 요약

### ✅ config.py
```python
# 추가된 설정들
lambda_ec: float = 0.1                    # EC consistency loss 가중치
lambda_rpn_value: float = 0.0             # RPN stack value regression (비활성화)
use_digit_conv: bool = False              # Digit-level 1D conv (비활성화)
use_plateau_schedule: bool = False        # Plateau scheduler (비활성화)
plateau_patience: int = 5                 # Plateau patience
plateau_factor: float = 0.5               # LR decay factor
```

**검증 결과**: ✅ 기본값이 안전하게 설정됨 (실험적 기능들은 모두 False/0.0)

---

### ✅ train.py

#### 1) EC Consistency Loss 추가
```python
def compute_ec_consistency_loss(...) -> torch.Tensor | None:
    # 동치 수식 그룹(group_id)의 출력 분포를 일치시키는 loss
    # Law Preservation / Expression Consistency 개선용
```

**검증 결과**: ✅ 정상 구현, `lambda_ec=0.1`로 활성화

#### 2) RPN Stack Value Supervision 추가
```python
def _compute_rpn_stack_values(expr: str) -> List[float]:
    # RPN 중간 계산 값을 scratchpad label로 제공
    # 음수 클램핑 제거됨 (데이터 생성에서 이미 방지)
```

**검증 결과**: ✅ 수정 완료, 음수 방어 로직 제거됨  
**이유**: 규칙(제3조 ②항)에 따라 음수 결과는 데이터 생성 단계에서 이미 필터링됨

#### 3) 카테고리별 Validation 샘플링 개선
```python
# 쉬운 케이스(덧셈/뺄셈) 대신 어려운 케이스 우선 표시
priority_categories = [
    "parentheses", "op3_plus", "large_number", "mixed",
    "identity", "division", ...
]
```

**검증 결과**: ✅ Hard-case 모니터링 강화

#### 4) Plateau Scheduler 옵션 추가
```python
if train_config.use_plateau_schedule:
    scheduler = ReduceLROnPlateau(optim, mode='max', ...)
```

**검증 결과**: ✅ 기본값 False로 비활성화됨

---

### ✅ dataloader.py

#### 1) Hard-case 생성 함수 강화
```python
# LP/Hard_LP: 괄호 우선순위 강화
# EC/Hard_EC: 교환/결합/분배 법칙 강화
# 큰수(4~7자리) 비율 증가
# 항등원(+0, *1) 패턴 추가
```

**검증 결과**: ✅ 벤치마크 약점(LP 19%, EC_HARD 0.5%) 직접 타겟팅

#### 2) `group_id` 메타 정보 추가
```python
# EC consistency를 위해 동치 수식 쌍에 동일한 group_id 부여
original_with_group["meta"]["group_id"] = group_id
aug_item["meta"]["group_id"] = group_id
```

**검증 결과**: ✅ EC consistency loss와 정확히 연동됨

---

### ✅ model.py

#### 1) Digit-level 1D Convolution 추가
```python
if self.use_digit_conv:
    self.digit_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
    # 자리올림 패턴 캡처용 (선택적)
```

**검증 결과**: ✅ `use_digit_conv=False`로 비활성화됨

#### 2) RPN Value Head 추가
```python
if use_rpn_value_head:
    self.rpn_value_out = nn.Linear(d_model, 1)
    # RPN stack 중간 값 예측용 (선택적)
```

**검증 결과**: ✅ `lambda_rpn_value=0.0`으로 비활성화됨

---

## 2. 규칙 준수 검증

### ✅ 제3조 ②항: 음수 결과 금지
- **이전**: `_compute_rpn_stack_values()`에서 `res < 0` 시 0으로 클램핑
- **현재**: 음수 클램핑 제거, 데이터 생성에서만 방지
- **이유**: 음수는 애초에 학습 데이터에 존재하지 않음 (규칙 준수)

### ✅ 제4조 ①항: 명시적 계산 금지
- `eval()`은 데이터 생성 시에만 사용 (train.py, dataloader.py)
- `Model.predict()`에는 `eval()` 없음
- **검증**: ✅ 규칙 준수

### ✅ 제5조 ④항: 사후 보정 금지
- 모든 loss는 train-time에만 사용
- predict()에는 어떤 보정도 없음
- **검증**: ✅ 규칙 준수

### ✅ 제7조: 입력 전처리 금지
- `input_text`는 원본 그대로 사용
- RPN은 auxiliary loss용으로만 사용 (입력이 아님)
- **검증**: ✅ 규칙 준수

---

## 3. 기본 설정 확인

### ✅ 활성화된 기능
```python
lambda_rpn = 0.2          # RPN auxiliary loss (기존 기능)
lambda_ec = 0.1           # EC consistency loss (신규 기능)
lr = 2e-4                 # Learning rate (최적화됨)
batch_size = 128          # Batch size (검증됨)
num_encoder_layers = 6    # Encoder depth (checkpoint 호환)
num_decoder_layers = 2    # Decoder depth (checkpoint 호환)
```

### ✅ 비활성화된 기능 (실험용)
```python
lambda_rpn_value = 0.0        # RPN stack value supervision
use_digit_conv = False        # Digit-level 1D convolution
use_plateau_schedule = False  # Plateau-based LR scheduler
```

---

## 4. 의도하지 않은 변경 검토

### ✅ 체크 항목

| 항목 | 상태 | 비고 |
|------|------|------|
| 기본 모델 구조 변경 없음 | ✅ | Transformer 구조 유지 |
| Checkpoint 호환성 유지 | ✅ | nhead=2, encoder=6, decoder=2 |
| RPN auxiliary loss 유지 | ✅ | lambda_rpn=0.2 그대로 |
| 기존 데이터 생성 로직 유지 | ✅ | Hard-case 추가만 함 |
| Validation 로직 유지 | ✅ | 샘플 표시 개선만 함 |
| 학습 루프 안정성 유지 | ✅ | Early stopping 유지 |

### ⚠️ 주의사항

1. **EC Consistency Loss**: 새로 추가된 loss이므로 첫 학습 시 모니터링 필요
   - `train/loss_ec` 값이 NaN이거나 너무 크면 `lambda_ec` 조정
   
2. **Group ID**: 증강 데이터에만 있으므로 base 데이터는 consistency loss 안 받음
   - 정상 동작임 (base + augmented 혼합)

3. **실험적 기능들**: 기본값으로 비활성화되어 있으나 필요시 활성화 가능
   - `use_digit_conv=True`: 큰수 곱셈 개선 목적
   - `lambda_rpn_value>0`: RPN 중간 값 supervision 추가
   - `use_plateau_schedule=True`: Validation 기반 LR 조정

---

## 5. Git Commit 준비

### 변경 파일 목록
```
modified:   config.py          (+31 lines: EC/RPN-value/digit-conv/plateau 설정 추가)
modified:   train.py           (+368 lines: EC loss, RPN value head, validation 개선)
modified:   dataloader.py      (+33 lines: hard-case 강화, group_id 추가)
modified:   model.py           (+47 lines: digit-conv, RPN value head 추가)
new:        RPN_EC_COMPARISON_REPORT.md  (문서)
```

### 권장 Commit 메시지
```
feat: Add EC consistency loss and hard-case curriculum

- Add EC consistency loss (lambda_ec=0.1) for Law/Expression Consistency
- Enhance hard-case data generation (LP/EC/large-number/identity)
- Add group_id metadata for equivalent expression pairs
- Remove unnecessary negative clamping in RPN stack values
- Add optional features (digit_conv, rpn_value_head, plateau_scheduler)
- Improve validation sampling to prioritize hard categories

Closes: Hard_EC (0.5% → target >20%), LP (19.7% → target >30%)
```

---

## 6. 최종 결론

### ✅ 모든 검토 항목 통과

1. ✅ 규칙 준수: 음수 클램핑 제거, 명시적 계산 없음
2. ✅ 기본 설정 안전: 실험적 기능 비활성화
3. ✅ Checkpoint 호환: nhead/encoder/decoder 유지
4. ✅ 의도한 개선: EC consistency, hard-case, validation
5. ✅ 의도하지 않은 변경: 없음

### 🚀 학습 준비 완료

모든 코드 변경이 안전하게 검증되었으며, 학습을 시작할 수 있습니다.

```bash
cd /Users/sunny/datathon/2025-inthon-baseline
source ../venv/bin/activate
python train.py
```

---

**검토 완료 일시**: 2025년 11월 16일  
**검토자**: AI Assistant  
**승인 상태**: ✅ 학습 시작 가능

