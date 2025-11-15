# EC Fine-tuning Guide

## 현재 상황 분석

### Baseline 스코어 (submission_inthon_sunwoo_ju)
```json
{
  "CA": 0.7757,  // Calculation Accuracy - 양호
  "LP": 0.7794,  // Law Preservation - 양호
  "EC": 0.2715,  // Expression Consistency - ⚠️ 문제!
  "RC": 0.8444   // Relational Consistency - 우수
}
```

### 문제 진단
- **EC가 27.15%로 매우 낮음** (다른 지표는 77-84%)
- EC 비중이 **30%**이므로, EC 향상 시 전체 점수 크게 개선
- CA/LP/RC는 이미 양호 → **EC 집중 파인튜닝**이 최적 전략

---

## 적용된 개선사항

### 1. 데이터 분포 조정 (평가 기반)

#### Before (Baseline):
```python
TRAINING_DISTRIBUTION = {
    "base_calculation": 0.40,
    "precedence": 0.20,
    "expression_consistency": 0.25,  # ⬅️ 문제
    "relational": 0.10,
    "single_number": 0.05,
}
```

#### After (EC-Focused):
```python
TRAINING_DISTRIBUTION = {
    "base_calculation": 0.25,        # 40% → 25% (CA 유지 가능)
    "precedence": 0.15,              # 20% → 15% (LP 유지 가능)
    "expression_consistency": 0.45,  # 25% → 45% ⬆️⬆️
    "relational": 0.10,              # RC 이미 우수, 유지
    "single_number": 0.05,           # baseline 유지
}
```

**근거**:
- EC를 45%로 올림 (80% 증가, 안정적 상한선)
- 50%는 과도할 위험 (CA/LP 동반 하락 우려)
- CA@77.57%, LP@77.94%는 25-30% 축소 허용 가능
- RC@84.44%는 이미 우수하여 유지

### 2. 대형 출력 비율 증가 (OOD EC 강화)

```python
OUTPUT_6DIGIT_RATIO = {
    "expression_consistency": 0.30,  # 0.20 → 0.30
}
```

**효과**:
- 큰 숫자 교환법칙 패턴 증가 (54321+98765 vs 98765+54321)
- OOD 테스트 대응력 향상
- 0.30은 안정성과 효과의 균형점

### 3. EC 전용 Augmentation 강화 (핵심!)

```python
# ArithmeticDataset.__getitem__ 내부
if category == "expression_consistency":
    augment_prob *= 1.5  # EC는 1.5배 boost
    augment_prob = min(augment_prob, 0.80)  # 최대 80%
```

**실제 augmentation 확률**:
| Phase | 기본 확률 | EC 카테고리 확률 | 증가율 |
|-------|-----------|------------------|--------|
| 1     | 10%       | 15%              | +50%   |
| 2     | 20%       | 30%              | +50%   |
| 3     | 35%       | 52%              | +49%   |
| 4     | 25%       | 37%              | +48%   |

**왜 효과적인가**:
- EC 샘플만 교환법칙/결합법칙 변형이 폭증
- A+B ↔ B+A, (A+B)+C ↔ A+(B+C) 노출 극대화
- **다른 카테고리는 영향 없음** → CA/LP/RC 능력 보존
- 코드 수정 최소, 효과 최대

### 4. EC 생성기 강화

`_gen_expression_consistency_base` 개선:
- 단순 이항 연산 비중: 60% → 75%
- 모든 패턴에서 50/50 순서 랜덤화
- 결합법칙도 50/50 괄호 위치

**효과**:
- A+B와 B+A가 정확히 균등하게 생성
- 모델이 순서 무관성을 강하게 학습

### 5. EC Probe Test 도구

**ec_probe.py**: EC 일관성 사전 검증 도구

```bash
python ec_probe.py best_model.pt        # Fine-tuning 전
python ec_probe.py best_model_ec.pt     # Fine-tuning 후
```

**측정 지표**:
- **EC Consistency**: (원본, 변형) 쌍이 모두 정답인 비율
- Original Accuracy: 원본 표현 정답률
- Augmented Accuracy: 변형 표현 정답률

**활용**:
- 공식 제출 전 EC 향상 확인
- Fine-tuning 효과 정량 측정
- A+B vs B+A 불일치 케이스 디버깅

---

## Fine-tuning 실행 방법

### 파일 복사
```bash
# 실행 환경으로 복사
cp config.py /path/to/colab/
cp model.py /path/to/colab/
cp train.py /path/to/colab/
cp dataloader.py /path/to/colab/
cp ec_probe.py /path/to/colab/  # EC 검증용
```

### 실행
```bash
# Fine-tuning (best_model.pt에서 시작)
python train.py

# EC 검증
python ec_probe.py best_model_ec.pt
```

### Fine-tuning 설정 (train.py)
```python
train_dataset = ArithmeticDataset(
    num_samples=300_000,  # 200k → 300k
    phase=2,              # 안정적인 2-3자리
    enable_augmentation=True,  # EC augmentation 활성화
)

train_config = TrainConfig(
    lr=1e-4,              # Fine-tuning용 낮은 LR
    warmup_steps=3000,    # 짧은 warmup
    num_epochs=10,        # 집중 학습
    save_best_path="best_model_ec.pt",
)
```

---

## 기대 효과 분석

### 정량적 예측

**EC 개선 경로**:
1. EC 샘플 비중: 25% → 45% (80% 증가)
2. EC augmentation: 20% → 30% (Phase 2 기준, 50% 증가)
3. 실효 EC 노출: 25% × 1.2 → 45% × 1.5 = **2.25배 증가**

**예상 결과**:
- EC: 27.15% → 42-48% (약 60-70% 상승)
- CA: 77.57% → 74-76% (소폭 하락)
- LP: 77.94% → 75-77% (소폭 하락)
- RC: 84.44% → 82-84% (미미한 변화)

**Weighted Score 계산** (CA:35%, LP:20%, EC:30%, RC:15%):

Before:
```
0.35×0.7757 + 0.20×0.7794 + 0.30×0.2715 + 0.15×0.8444
= 0.2715 + 0.1559 + 0.0815 + 0.1267
= 0.6356 (63.56%)
```

After (Conservative Estimate: EC=42%, CA=75%, LP=76%, RC=83%):
```
0.35×0.75 + 0.20×0.76 + 0.30×0.42 + 0.15×0.83
= 0.2625 + 0.1520 + 0.1260 + 0.1245
= 0.6650 (66.50%)
```

**Net Gain: +2.94% → 약 3% 향상!**

### 정성적 평가

**장점**:
1. ✅ **최소 침습**: 기존 코드 구조 유지
2. ✅ **타겟 명확**: EC만 집중, 다른 능력 보존
3. ✅ **검증 가능**: ec_probe.py로 사전 확인
4. ✅ **리스크 관리**: 45% 상한, Phase 2 유지

**단점**:
1. ⚠️ CA/LP 소폭 하락 가능성
2. ⚠️ EC 45%는 여전히 공격적 (40%가 더 안전할 수도)

**완화 방안**:
- Fine-tuning LR을 1e-4로 낮춤 → 기존 능력 보존
- Warmup 3000으로 짧게 → 빠른 수렴
- 10 epoch로 제한 → 과적합 방지

---

## 체크리스트

### 실행 전
- [ ] `dataloader.py` EC 비중 45% 확인
- [ ] `train.py` lr=1e-4, epochs=10 확인
- [ ] `best_model.pt` 존재 확인
- [ ] Phase 2 설정 확인

### 실행 중
- [ ] WandB에서 EC augmentation 동작 확인
- [ ] valid/EM이 0.7+ 유지되는지 모니터링
- [ ] Early stopping 없이 10 epoch 완주

### 실행 후
- [ ] `ec_probe.py best_model_ec.pt` 실행
- [ ] EC Consistency 40%+ 달성 확인
- [ ] 공식 제출 및 리더보드 확인

---

## Troubleshooting

### Q: CA/LP가 너무 떨어지면?
A: EC 비중을 45% → 40%로 낮추고 재학습

### Q: EC가 40% 안 넘으면?
A: 
1. augment_prob 배수를 1.5 → 2.0으로 증가
2. 이항 연산 비중을 75% → 85%로 증가
3. num_samples를 300k → 500k로 증가

### Q: 과적합 의심되면?
A: 
1. dropout을 0.0 → 0.1로 증가
2. weight_decay를 0.1 → 0.2로 증가
3. epochs를 10 → 7로 감소

---

## 결론

**이번 EC fine-tuning 전략은**:
- 철저한 스코어 분석 기반
- 최소 변경, 최대 효과
- 검증 도구 포함
- 리스크 관리됨

**예상 성과**: EC 27% → 42-48%, 전체 점수 +3% 향상

