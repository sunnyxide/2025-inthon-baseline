# A100 Depth Expansion Report

## 1. Baseline reference (best_model.pt)

| Category | Value | Notes |
|----------|-------|-------|
| d_model | 256 | Stored in `best_model.pt`, validated via `ModelConfig` comments |
| nhead | 2 | W&B sweep 최적값 |
| num_encoder_layers | 6 | 기존 Transformer encoder 깊이 |
| num_decoder_layers | 2 | 기존 Transformer decoder 깊이 |
| dim_feedforward | 1024 | Feed-forward hidden size |
| dropout | 0.0 | sweep 결과 과적합 이슈 없음 |
| lr | 5e-4 | best sweep 결과 |
| warmup_steps | 5000 | baseline schedule |
| weight_decay | 0.1 | 문헌 권장 |
| grad_clip | 1.0 | 안정화 |
| valid_every | 200 | baseline validation 주기 |
| num_epochs | 20 | sweep 결과 |
| batch_size | 128 | 원본 제출 baseline과 동일 |
| dataset_size | 900k train / 3k val | phase mix (2-4) train / phase 4 val |

## 2. Compatibility constraints

1. `d_model`, `nhead`, `dim_feedforward`는 checkpoint와 동일해야 shape mismatch 없이 로드 가능함.
2. Encoder/decoder layer 수를 늘릴 경우 `load_state_dict(strict=False)` 로딩이 필요하며, 추가 layer는 새로 초기화됨.
3. Batch size를 128로 유지하여 baseline 파이프라인을 그대로 활용.

## 3. Observations

- 기존 best hyperparameters는 작은 모델에서도 안정적이었으므로, depth 확장 시에도 그대로 사용하는 것이 합리적.
- Depth만 늘리는 전략은 checkpoint 호환성을 해치지 않으며, 추가 layer 초기 가중치만 새로 초기화하면 됨.
- 데이터셋 800k/2k 구성은 이미 설정되어 있으므로, training pipeline 변경 없이 적용 가능.

## 4. Depth expansion candidates (batch_size=512 고정)

| Profile | Encoder / Decoder Layers | Relative Params | 장점 | 주의사항 |
|---------|-------------------------|-----------------|------|-----------|
| `LEGACY_POWERUP` | 8 / 3 | +33% | shallow 대비 추론 깊이 증가, latency 증가 적음 | checkpoint 로드시 2 encoder layer/1 decoder layer가 새로 초기화됨 |
| `DEEP_CONTEXT` | 10 / 4 | +66% | 긴 dependency 처리, validation 성능 기대 | 학습 step 당 latency 약 +20% |
| `ULTRA_CONTEXT` | 12 / 4 | +100% | 복잡 식 처리, A100 활용 극대화 | 추가 4 encoder layer random init → 초기 안정화 warmup 필요 |

공통 사양:
- `d_model=256`, `nhead=2`, `dim_feedforward=1024`, `dropout=0.0`
- LR/스케줄: lr=5e-4, warmup=5000, cosine decay 유지
- batch_size=128, 데이터셋 900k(train) / 3k(val), phase mix(2-4)
]
선호안:
1. **LEGACY_POWERUP**: 구조 변경이 최소이며 baseline 대비 2 encoder / 1 decoder layer 증가.
2. **DEEP_CONTEXT**: GPU 여력이 있다면 추천. 추가 layer가 많지만 여전히 파라미터 ~5M 수준.
3. **ULTRA_CONTEXT**: 연구용. Early epoch에서 gradient explosion 감시 필요 (grad_clip 유지).

## 5. Data diversity upgrades

- Training generator에 `long_expression`, `complex_nested` 카테고리 추가 (총 7개 분포).
- Phase mix `(2, 3, 4)`을 샘플마다 무작위로 선택하여 자리수 다양화.
- `_gen_base_calculation`의 긴 수식 비중을 12%로 확대.
- `phase_mix` 파라미터를 `ArithmeticDataset`에 도입하여 DataLoader 수준에서 난이도 교차를 지원.

## 6. RPN auxiliary head 타당성 평가

- **장점**
  - 동일 encoder representation을 공유하면서, 연산 순서를 명시적으로 학습해 긴 연산/괄호 조합의 일반화를 개선할 가능성이 높음.
  - depth_profile로 늘어난 디코더 용량을 RPN head에도 재사용하므로 파라미터 증가가 제한적 (projection 한 층 추가 수준).
  - inference 경로는 기존 `generate()`/`predict()`를 그대로 사용하므로 제출 규칙을 위반하지 않음.
- **위험/대응**
  - best_model 기반 strict 로딩 시 새 디코더 레이어와 RPN head는 새로 초기화되므로, 초기 수렴을 위해 λ 조절이 필수 → 기본값 0.2 제안.
  - RPN 타깃 생성 로직(infix→RPN)이 오류를 내면 전체 학습이 망가지므로, 괄호/나눗셈 처리, 연속 숫자 파싱 등을 유닛 테스트로 검증해야 함.
  - 학습 메모리 사용량이 증가하므로 batch size 128 유지 권장. 필요 시 GradAccum 옵션 대비.
