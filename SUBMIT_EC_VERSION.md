# EC Fine-tuning 버전 제출 가이드

## 📦 브랜치: `feature/ec-focus`

성공했던 baseline을 기반으로 **EC (Expression Consistency) 27.15% → 40-50% 향상**에 집중한 버전입니다.

---

## 🎯 목표 및 전략

### 현재 스코어 분석
```
CA (Calculation Accuracy):    77.57% ✅ (비중 35%)
LP (Law Preservation):         77.94% ✅ (비중 20%)
EC (Expression Consistency):   27.15% ❌ (비중 30%) ⬅️ 타겟!
RC (Relational Consistency):   84.44% ✅ (비중 15%)
```

### 전략
- **EC만 집중 강화** (27% → 42-48% 목표)
- CA/LP/RC 능력 최대한 보존
- Fine-tuning 방식으로 안정적 개선

---

## 📋 복사할 파일 (5개)

### 필수 파일
```bash
/Users/sunny/datathon/2025-inthon-baseline/
├── config.py           # Baseline 설정 (show_valid_samples=10)
├── model.py            # 깔끔한 TransformerSeq2Seq (RPN 없음)
├── train.py            # EC fine-tuning 설정
├── dataloader.py       # EC 45% + augmentation 강화
└── ec_probe.py         # EC 검증 도구
```

### 실행 환경으로 복사
```bash
# Google Colab/Kaggle 등 실행 환경의 작업 폴더에 복사
cp config.py /content/drive/MyDrive/submission/
cp model.py /content/drive/MyDrive/submission/
cp train.py /content/drive/MyDrive/submission/
cp dataloader.py /content/drive/MyDrive/submission/
cp ec_probe.py /content/drive/MyDrive/submission/

# best_model.pt도 함께 복사 필수!
cp best_model.pt /content/drive/MyDrive/submission/
```

---

## 🚀 실행 방법

### 1. Fine-tuning 실행
```bash
cd /content/drive/MyDrive/submission/
python train.py
```

### 2. EC 검증 (선택사항)
```bash
# Fine-tuning 전 EC 측정
python ec_probe.py best_model.pt

# Fine-tuning 후 EC 측정
python ec_probe.py best_model_ec.pt
```

### 3. 최종 제출
- `best_model_ec.pt` 사용
- 또는 validation EM이 더 높은 모델 선택

---

## 🔧 핵심 개선사항

### 1. EC 데이터 비중 강화
```python
# dataloader.py
TRAINING_DISTRIBUTION = {
    "expression_consistency": 0.45,  # 25% → 45% (⬆️ 80%)
    "base_calculation": 0.25,        # 40% → 25%
    "precedence": 0.15,              # 20% → 15%
    ...
}
```

### 2. EC 전용 Augmentation Boost (핵심!)
```python
# dataloader.py, __getitem__ 내부
if category == "expression_consistency":
    augment_prob *= 1.5  # EC 샘플만 1.5배
```

**Phase 2 기준 실효 augmentation**:
- 다른 카테고리: 20%
- **EC 카테고리: 30%** (1.5배)

### 3. 큰 수 연산 학습 강화
```python
# dataloader.py
OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.15,        # 0.05 → 0.15 (3배)
    "precedence": 0.20,              # 0.10 → 0.20 (2배)
    "expression_consistency": 0.35,  # 0.20 → 0.35
    "relational": 0.40,              # 0.30 → 0.40
}
```

**효과**:
- 5-6자리 결과 생성 대폭 증가
- OOD 큰 수 일반화 능력 향상
- EC with large numbers 강화

### 4. EC 생성기 강화
```python
# _gen_expression_consistency_base
- 단순 이항 연산: 60% → 75%
- 순서 랜덤화: 모든 패턴에서 50/50
- 교환법칙 A+B vs B+A 균등 생성
- 결합법칙 (A+B)+C vs A+(B+C) 균등 생성
```

### 5. 카테고리별 Validation 표시
- 10개 카테고리 균형있게 표시
- 교환법칙, 괄호, 큰수, 혼합, 0/1 포함 등
- 약점 즉시 파악 가능

---

## 📊 Fine-tuning 설정

### train.py (main 함수)
```python
train_dataset = ArithmeticDataset(
    num_samples=300_000,  # 200k → 300k (50% 증가)
    phase=2,              # 안정적인 2-3자리
    enable_augmentation=True,
)

train_config = TrainConfig(
    lr=1e-4,              # Fine-tuning용 낮은 LR
    warmup_steps=3000,    # 빠른 수렴
    num_epochs=10,        # 집중 학습
    save_best_path="best_model_ec.pt",
)
```

### 왜 이 설정인가?
- **LR 1e-4**: 기존 능력 보존하면서 EC만 개선
- **Warmup 3000**: Fine-tuning이므로 짧게
- **Epochs 10**: 과적합 방지, 집중 학습
- **Samples 300k**: EC 노출 충분히 확보

---

## 📈 예상 결과

### Conservative Estimate
```
Before: CA=77.57%, LP=77.94%, EC=27.15%, RC=84.44%
After:  CA=75.00%, LP=76.00%, EC=42.00%, RC=83.00%

Weighted Score: 63.56% → 66.50% (+2.94%)
```

### Optimistic Estimate  
```
Before: CA=77.57%, LP=77.94%, EC=27.15%, RC=84.44%
After:  CA=76.00%, LP=77.00%, EC=48.00%, RC=83.00%

Weighted Score: 63.56% → 68.30% (+4.74%)
```

### 핵심 포인트
- **EC 개선이 가장 중요** (비중 30%, 현재 27%로 최약점)
- EC 15-20% 향상 = 전체 점수 4.5-6% 향상
- CA/LP 2-3% 하락은 허용 가능 범위

---

## ✅ 실행 전 체크리스트

- [ ] `best_model.pt` 파일 존재 확인
- [ ] 5개 파일 모두 복사 완료
- [ ] `train.py`에서 `resume_from_checkpoint=True` 확인
- [ ] `train.py`에서 `lr=1e-4` 확인
- [ ] WandB 로그인 완료

---

## 📊 모니터링 포인트

### 학습 중
- `valid/EM` 0.70+ 유지되는지 확인
- `train/lr`이 1e-4에서 시작하는지 확인
- Epoch마다 카테고리별 샘플 10개 확인

### Validation 출력 예시
```
Sample Validation Output (카테고리별 10개):
================================================================================
  [ 0] ✓ [교환법칙        ] | input: 23+45                       | target: 68           | pred: 68
  [ 1] ✓ [괄호연산        ] | input: (12+34)*5                   | target: 230          | pred: 230
  [ 2] ✓ [큰수연산(5+자리)] | input: 9999*123                    | target: 1229877      | pred: 1229877
  [ 3] ✓ [혼합연산        ] | input: 12+34*56                    | target: 1916         | pred: 1916
  [ 4] ✓ [덧셈            ] | input: 123+456                     | target: 579          | pred: 579
  [ 5] ✓ [곱셈            ] | input: 45*67                       | target: 3015         | pred: 3015
  [ 6] ✓ [뺄셈            ] | input: 100-23                      | target: 77           | pred: 77
  [ 7] ✓ [나눗셈          ] | input: 100//5                      | target: 20           | pred: 20
  [ 8] ✓ [0포함연산       ] | input: 45+0                        | target: 45           | pred: 45
  [ 9] ✓ [1포함연산       ] | input: 67*1                        | target: 67           | pred: 67
================================================================================
```

---

## 🧪 EC Probe 사용법

### EC 일관성 측정
```bash
python ec_probe.py best_model_ec.pt
```

### 출력 예시
```
🔬 EC (Expression Consistency) Probe Test
======================================================================
📊 EC Probe Results
======================================================================
Total pairs:        500
Both correct:       410 (82.0%)
Orig only:           45 (9.0%)
Aug only:            30 (6.0%)
Both wrong:          15 (3.0%)

🎯 EC Consistency:  82.00%
   (= both expressions give same answer)

📈 Original accuracy:   91.00%
📈 Augmented accuracy:  88.00%
======================================================================
```

**해석**:
- **EC Consistency 82%**: A+B와 B+A 모두 같은 답 (목표!)
- Original/Aug 차이가 작을수록 좋음

---

## ⚠️ Troubleshooting

### Q1: EC가 40% 안 넘으면?
**A**: 
1. `dataloader.py`에서 augment_prob 배수를 1.5 → 2.0으로 증가
2. 이항 연산 비중 75% → 85%로 증가
3. num_samples 300k → 500k로 증가

### Q2: CA/LP가 75% 아래로 떨어지면?
**A**:
1. EC 비중을 45% → 40%로 축소
2. base_calculation을 25% → 30%로 복원
3. LR을 1e-4 → 5e-5로 더 낮춤

### Q3: 큰 수 연산이 잘 안되면?
**A**:
1. OUTPUT_6DIGIT_RATIO를 더 올림 (현재 0.15-0.40 → 0.25-0.50)
2. Phase 3로 학습 (3-4자리 → 더 큰 숫자)
3. max_gen_len을 50 → 64로 증가

---

## 📝 최종 요약

### 핵심 변경사항
1. ⭐⭐⭐⭐⭐ **EC 전용 Augmentation 1.5배**
2. ⭐⭐⭐⭐⭐ **EC 데이터 비중 45%**
3. ⭐⭐⭐⭐ **큰 수 연산 3-4배 강화**
4. ⭐⭐⭐⭐ **카테고리별 Validation 10개**
5. ⭐⭐⭐⭐⭐ **EC Probe 검증 도구**

### 예상 성과
- **EC**: 27.15% → 42-48% (+55-77% 향상)
- **전체 점수**: 63.56% → 66.50-68.30% (+3-5% 향상)

### 리스크 관리
- Conservative 45% EC ratio (not 50%)
- Fine-tuning LR (not retraining)
- Phase 2 유지 (안정성)
- 검증 도구로 사전 확인

---

## 🚀 실행 커맨드

```bash
# 파일 복사
cp config.py model.py train.py dataloader.py ec_probe.py best_model.pt /path/to/execution/

# Fine-tuning
cd /path/to/execution/
python train.py

# EC 검증
python ec_probe.py best_model_ec.pt

# 제출
# best_model_ec.pt를 제출 폴더에 배치
```

---

**모든 준비 완료! 실행 후 EC 향상을 확인하세요.** 🎯

