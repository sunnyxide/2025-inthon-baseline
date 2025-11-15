## RPN vs Attention Baseline Comparison (EC / Hard Cases Focus)

> Developer log: Summary document for comparing the improved RPN-based model against the previous attention baseline and legacy RPN model.  
> 실제 수치는 학습 이후 `evaluate_arithmetic.py`와 공식 벤치마크로 갱신할 수 있도록 구조만 고정합니다.

### 1. Final RPN Model Configuration (After EC & Hard-Case Improvements)

- **ModelConfig**
  - `d_model=256`
  - `nhead=2`
  - `num_encoder_layers`: profile별 (`baseline`, `legacy_powerup`, `deep_context`)
  - `num_decoder_layers`: profile별 설정
  - `dim_feedforward=1024`
  - `dropout=0.0`
  - `use_digit_conv`: optional 1D conv over decoder time dimension (carry pattern modeling)
- **TrainConfig (핵심)**
  - `lr ≈ 2e-4`, warmup `≈ 7k`, cosine schedule (`min_lr_threshold=1e-5`)
  - `batch_size=128` (또는 256 sweep)
  - `max_train_steps ≈ 90k~120k`
  - `lambda_rpn=0.2` (RPN token loss)
  - `lambda_ec=0.1` (expression-consistency loss for EC 동치 수식 그룹)
  - `lambda_rpn_value` (RPN stack-value regression loss, default 0.0 → 실험 시 >0로 설정)
  - Depth profile sweep: `baseline / legacy_powerup / deep_context`

- **Data curriculum (요약)**
  - `TRAINING_DISTRIBUTION` 조정으로 **precedence / law_preservation / expression_consistency / relational / complex_nested** 비중 강화
  - `OUTPUT_6DIGIT_RATIO` 조정으로 **큰수(6+ 자리) 출력** 비율 증가
  - `create_augmented_dataset_from_original`에서
    - 괄호/법칙/관계성 카테고리에 대해 `max_augmentations` 증가
    - 모든 동치 수식에 `group_id` 부여 → EC consistency loss와 pair 분석에 사용

### 2. RPN Scratchpad (Stack-Value) Supervision

- 학습 시 `train.py`에서:
  - Infix 식을 `_infix_to_rpn_numbers()`로 변환 후 **숫자 단위 RPN 시퀀스** 생성
  - 각 연산자 적용 직후 스택 top 값을 `_compute_rpn_stack_values()`로 계산
    - 결과는 `log10(value + 1)` 형태로 스케일링
  - char-level RPN 토큰에서 `+, -, *, D(//)` 위치에만 해당 값을 매핑하고, 나머지 위치는 마스크
- 모델 쪽(`TransformerSeq2Seq.forward_with_rpn`):
  - RPN 디코더 hidden (`rpn_out`)에 대해
    - `rpn_out` → `rpn_out_linear` (기존 토큰 예측)
    - `rpn_out` → `rpn_value_out` (stack-value 회귀, shape: `[B, T_rpn]`)
- Loss:
  - `loss_rpn`: RPN 토큰 CE
  - `loss_rpn_value`: 마스크된 위치에 대해 MSE (`lambda_rpn_value`로 가중)

### 3. Benchmark Metrics to Track

#### 3-1. Public Benchmark (기존 결과 요약)

| Model                | Calc Acc | Law Pres | Expr Cons | Rel Cons | 기타(0.6355 등) |
|----------------------|---------:|---------:|----------:|---------:|----------------:|
| Attention baseline   |   35%    |   20%    |    30%    |   15%    | 0.6355 / 0.7757 / 0.7794 / 0.2715 / 0.8444 |
| Legacy RPN (이전 run) | (작성자 기록) | (작성자 기록) | (작성자 기록) | (작성자 기록) | (필요 시 추가) |
| **Improved RPN (현재)** | (TODO)  | (TODO)  | (TODO)    | (TODO)  | (TODO)         |

> 실제 점수는 학습 완료 후 공개 벤치마크에 제출하여 위 표에 채우면 됩니다.

#### 3-2. Local Evaluation (`evaluate_arithmetic.py`)

현재 코드 기준 예시 (작성 시점의 RPN 개선 모델):

- **Regular**: 39.07% (Weighted 39.58%)
  - CA: 46.76%, EC: 48.25%, LP: 19.73%, RC: 31.95%
- **OOD**: 23.00%
- **Hard_EC**: 0.50%
- **Hard_LP**: 0.00%
- **Hard_RC**: 25.50%

향후 학습/파인튜닝 후에는 동일 스크립트를 재실행하여:

1. 개선된 RPN 모델의 위 수치를 다시 측정하고  
2. Attention baseline / legacy RPN 결과와 함께 표에 정리하면  
3. Law/Expression/Relational Consistency 측면에서의 향상을 한눈에 비교할 수 있습니다.

### 4. 비교·보고용 체크리스트

- [ ] 동일한 `evaluate_arithmetic.py` 버전으로 **Attention baseline**, **Legacy RPN**, **Improved RPN** 모두 측정
- [ ] 각 모델에 대해 Regular/OOD/Hard_EC/Hard_LP/Hard_RC accuracy 기록
- [ ] Public benchmark에서 Calc/Law/Expr/Relational Consistency 점수 비교
- [ ] 최종 선택 모델과 설정(`TrainConfig`, `ModelConfig`, depth_profile, `lambda_*` 값)을 `FINAL_REVIEW_REPORT.md`에 요약 반영


