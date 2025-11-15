# 실행 시 복사해야 할 파일 목록

## ✅ 업데이트된 파일 (반드시 복사 필요)

### 1. **config.py** ⭐ 필수
- `lambda_rpn` 파라미터 추가 (RPN auxiliary loss weight)
- `DEPTH_PROFILES` 및 `apply_depth_profile()` 함수
- `TrainConfig`에 depth_profile, train_num_samples, val_num_samples, train_phase_mix, val_phase 추가

### 2. **model.py** ⭐ 필수
- `TransformerSeq2Seq`에 RPN 디코더 + 전용 임베딩/출력(`self.rpn_embed`, `self.rpn_out`) 추가
- `forward_with_rpn()` 메서드 추가 (훈련 시 RPN 보조 학습용; `rpn_vocab` 없으면 예외)
- 기존 `forward()`, `generate()`, `Model.predict()`는 그대로 유지 (추론 경로 변경 없음)

### 3. **train.py** ⭐ 필수
- `infix_to_rpn()` 함수 추가 (infix → RPN 변환)
- `build_rpn_tokenizer()` 함수 추가 (RPN용 토크나이저 생성)
- `_encode_rpn_text()` / `_pad_sequences()` 유틸 추가 (문자 집합 검증 + 패딩)
- `train_loop()`에 RPN 학습 로직 추가 (joint loss: `loss + lambda_rpn * loss_rpn`)
- `main()` 및 `train_run()`에서 RPN 토크나이저 생성 + `rpn_vocab` 전달

### 4. **dataloader.py** ⭐ 필수
- `long_expression` 카테고리 추가 (10%)
- `complex_nested` 카테고리 추가 (10%)
- `_gen_long_expression()` 함수 추가
- `_gen_complex_nested()` 함수 추가
- `ArithmeticDataset`에 `phase_mix` 지원 추가
- 긴 수식 생성 확률 증가 (5% → 12%)

---

## 📋 전체 파일 목록 (실행 환경에 복사)

```
필수 파일:
├── config.py          ✅ RPN + depth profile 설정
├── model.py           ✅ RPN 디코더 추가
├── train.py           ✅ RPN 학습 루프
├── dataloader.py      ✅ 데이터 다양화 (long_expression, complex_nested)
│
기존 파일 (변경 없음):
├── do_not_edit/       (그대로 유지)
│   ├── model_template.py
│   ├── metric.py
│   └── dataloader_validator.py
│
선택 파일:
└── DEPTH_EXPANSION_REPORT.md  (참고용 문서)
```

---

## 🚀 실행 방법

### 1. 파일 복사
위 4개 파일(`config.py`, `model.py`, `train.py`, `dataloader.py`)을 실행 환경에 복사합니다.

### 2. 실행
```bash
python train.py
```

### 3. RPN 비활성화 (선택사항)
RPN을 사용하지 않으려면 `config.py`에서:
```python
lambda_rpn: float = 0.0  # 0.2 → 0.0으로 변경
```
또는 `train.py`의 `main()`에서:
```python
rpn_tokenizer = None  # build_rpn_tokenizer() 대신 None
```

---

## ⚠️ 주의사항

### 체크포인트 호환성
- 기존 `best_model.pt` (d_model=256, nhead=2)는 **호환됨**
- RPN 디코더/임베딩은 새로 초기화되므로 자동으로 `strict=False` 로드 (로그에 missing key 안내)
- depth_profile로 layer 수를 늘리면 추가 layer만 새로 초기화됨

### 메모리 사용량
- RPN 디코더 추가로 약 **+30% 메모리** 사용
- batch_size=128 기준으로 A100에서 충분히 실행 가능

### 학습 설정
- 기본값: `lambda_rpn = 0.2` (RPN loss 가중치)
- `lambda_rpn = 0.0`으로 설정하면 RPN 비활성화 (기존 모델과 동일)
- `lambda_rpn > 0`일 때만 RPN 디코더/임베딩이 활성화되며, 토크나이저는 {0-9, 공백, +, -, *, D}만 사용

---

## 📊 변경 사항 요약

| 파일 | 주요 변경사항 |
|------|-------------|
| **config.py** | `lambda_rpn`, depth_profile, 대규모 데이터셋 설정 추가 |
| **model.py** | RPN 디코더 + `forward_with_rpn()` 추가 |
| **train.py** | RPN 타깃 생성 + joint loss + 안전한 토큰 검증 추가 |
| **dataloader.py** | long_expression, complex_nested 카테고리 추가 |

---

## ✅ 검증 체크리스트

실행 전 확인:
- [ ] `config.py`에 `lambda_rpn` 필드 있음
- [ ] `model.py`에 `forward_with_rpn()` 메서드 있음
- [ ] `train.py`에 `infix_to_rpn()` 함수 있음
- [ ] `dataloader.py`에 `_gen_long_expression()`, `_gen_complex_nested()` 함수 있음

실행 후 확인:
- [ ] 학습 시작 시 "RPN head enabled" 또는 유사 메시지 출력
- [ ] WandB에 `train/loss_rpn` 메트릭 기록됨 (lambda_rpn > 0일 때)
- [ ] 메모리 사용량이 기존 대비 약 30% 증가 (정상)

