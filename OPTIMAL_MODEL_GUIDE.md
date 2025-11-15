# 최적 모델 구현 가이드 (A to Z)

> **목표**: 규칙을 지키면서 산술 사칙연산 성능을 극대화하고, 교환/결합법칙, 괄호, OOD, 기호연산 등 다양한 평가를 수행할 수 있는 모델 구현

---

## 📋 목차

1. [전략 개요](#전략-개요)
2. [아키텍처 설계](#아키텍처-설계)
3. [학습 전략](#학습-전략)
4. [데이터 전략](#데이터-전략)
5. [구현 단계별 가이드](#구현-단계별-가이드)
6. [변경 파일 목록](#변경-파일-목록)

---

## 전략 개요

### 핵심 개선 사항

1. **Attention 메커니즘 도입** ⭐ (최우선)
   - Bahdanau Attention 또는 Multi-Head Attention
   - 입력 시퀀스의 모든 타임스텝 활용
   - 긴 수식 처리 능력 향상

2. **모델 용량 증가**
   - `d_model`: 256 → 512 또는 768
   - `num_layers`: 1 → 2~3
   - 더 강력한 표현력 확보

3. **Beam Search 도입**
   - Greedy Decoding → Beam Search (beam_size=3~5)
   - 전체 시퀀스 관점에서 최적화

4. **Scheduled Sampling**
   - Teacher Forcing 비율 점진적 감소
   - 학습-추론 분포 차이 완화

5. **Auxiliary Loss**
   - 연산자 우선순위 학습
   - 중간 계산 단계 학습 (규칙 허용 범위 내)

6. **Curriculum Learning**
   - 쉬운 수식부터 어려운 수식으로 점진적 학습
   - 일반화 성능 향상

7. **데이터 증강**
   - 교환법칙: `A+B` → `B+A`
   - 결합법칙: `(A+B)+C` → `A+(B+C)`
   - 괄호 추가/제거 (수학적으로 동등한 경우)

---

## 아키텍처 설계

### 1. Attention 기반 Seq2Seq 모델

#### 구조 개요

```
입력 수식: "3335+472"
    ↓
[CharTokenizer] → 문자 단위 토큰화
    ↓
[Embedding + Positional Encoding] → d_model 차원 벡터
    ↓
[Encoder (Multi-layer GRU)] → 모든 타임스텝의 hidden states
    ↓
[Bahdanau Attention] → 디코더가 인코더 출력에 집중
    ↓
[Decoder (Multi-layer GRU + Attention)] → hidden state 생성
    ↓
[Linear Projection] → vocab 크기 logits
    ↓
[Beam Search] → 최적 시퀀스 선택
    ↓
출력 숫자: "3807"
```

#### 주요 구성 요소

1. **Encoder**
   - Multi-layer GRU (2~3 layers)
   - Bidirectional (선택사항)
   - 모든 타임스텝의 hidden states 보존

2. **Attention Mechanism**
   - Bahdanau Attention (Additive)
   - 또는 Multi-Head Attention (Transformer 스타일)
   - 디코더가 인코더의 특정 부분에 집중

3. **Decoder**
   - Multi-layer GRU (2~3 layers)
   - Attention context를 입력으로 사용
   - Teacher Forcing + Scheduled Sampling

4. **Output Layer**
   - Linear projection
   - Softmax (학습 시)
   - Beam Search (추론 시)

### 2. 하이퍼파라미터 설정

```python
# ModelConfig
d_model = 512              # 256 → 512 (용량 증가)
num_encoder_layers = 2     # 1 → 2 (다층 구조)
num_decoder_layers = 2     # 1 → 2
num_attention_heads = 8    # Multi-Head Attention (선택)
dropout = 0.1              # 정규화
attention_type = "bahdanau" # 또는 "multihead"

# TrainConfig
learning_rate = 1e-3       # 2e-3 → 1e-3 (안정적 학습)
batch_size = 128           # 유지
num_epochs = 50            # 유지
max_gen_len = 32           # 유지
beam_size = 5              # Beam Search 크기
teacher_forcing_ratio = 0.9 # 초기값, 점진적 감소
```

---

## 학습 전략

### 1. Loss Function

#### Primary Loss
- Cross-Entropy Loss (기존과 동일)
- 시퀀스 전체에 대한 평균

#### Auxiliary Loss (선택사항)
- 연산자 위치 예측
- 중간 계산 단계 예측 (규칙 허용 범위 내)
- 가중치: `alpha * primary_loss + beta * auxiliary_loss`

### 1-1. RPN Auxiliary Decoder (depth profile + λ<sub>RPN</sub>)

- **목적**: 인픽스 수식을 직접 바꾸지 않고, 학습 과정에서만 RPN(Reverse Polish Notation) 시퀀스를 예측하도록 하여 연산 우선순위/괄호 패턴에 대한 일반화를 강화.
- **구성**
  - Encoder는 depth profile(`baseline`, `LEGACY_POWERUP`, `DEEP_CONTEXT`, `ULTRA_CONTEXT`)로 확장된 Transformer를 그대로 사용.
  - Result decoder와 동일한 depth를 갖는 **보조 디코더**를 추가하고, 동일한 char vocab을 사용해 RPN 문자열을 생성.
  - Inference(`predict()`)에서는 기존 경로만 사용하므로 제출 규칙에 영향을 주지 않음.
- **학습**
  - `loss = CE(result) + λ_rpn * CE(rpn)` 형태의 joint loss.
  - `TrainConfig.lambda_rpn` (기본 0.2)로 보조 loss 비중 조절.
  - Batch별로 infix → RPN 변환 후 BOS/EOS를 붙여 입력/정답 시퀀스를 구성.
  - Batch size 128, `train_num_samples=900k`, `train_phase_mix=(2,3,4)` 설정과 함께 사용하면 장·복합 수식에서 안정적인 수렴 확인.
- **주의**
  - 기존 `best_model.pt`를 strict=False로 로드하여 RPN head 및 추가 디코더 레이어 파라미터를 새로 초기화해야 함.
  - RPN 토크나이저는 학습 시에만 사용하며, 공백/연산자를 포함하도록 별도로 구성.

### 2. Scheduled Sampling

```python
# Teacher Forcing 비율을 점진적으로 감소
teacher_forcing_ratio = max(0.5, 1.0 - step / total_steps * 0.5)
```

### 3. Learning Rate Schedule

```python
# Warmup + Cosine Annealing
warmup_steps = 1000
max_lr = 1e-3
min_lr = 1e-5
```

### 4. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 데이터 전략

### 1. Curriculum Learning

```python
# 단계별 난이도 증가
epochs_1_10:   max_depth=2, num_digits=(1,3)
epochs_11_30:  max_depth=3, num_digits=(1,4)
epochs_31_50:  max_depth=4, num_digits=(1,5)  # 검증과 동일
```

### 2. 데이터 증강

#### 교환법칙 (Commutative Law)
```python
"A+B" → "B+A"  # 덧셈, 곱셈
"A*B" → "B*A"
```

#### 결합법칙 (Associative Law)
```python
"(A+B)+C" → "A+(B+C)"  # 덧셈, 곱셈
"(A*B)*C" → "A*(B*C)"
```

#### 분배법칙 (Distributive Law)
```python
"A*(B+C)" → "A*B+A*C"
```

#### 괄호 추가/제거 (수학적으로 동등한 경우)
```python
"A+B+C" → "(A+B)+C"  # 덧셈은 결합법칙 성립
"A*B*C" → "(A*B)*C"  # 곱셈은 결합법칙 성립
```

**⚠️ 주의**: 규칙에 따라 입력 텍스트를 변경하는 것은 금지되지만, 학습 데이터 생성 시 이러한 변형을 포함하는 것은 허용됩니다.

### 3. OOD 대비

- 학습 데이터: 1~5자리 숫자
- 검증 데이터: 1~5자리 (일부 6자리 이상 포함)
- 큰 숫자 처리 연습을 위한 데이터 포함

---

## 구현 단계별 가이드

### Phase 1: 기본 Attention 도입 (최우선)

**목표**: Attention 메커니즘을 도입하여 기본 성능 향상

**변경 파일**:
- `model.py`: `TinySeq2Seq` → `AttentionSeq2Seq`
- `config.py`: Attention 관련 설정 추가

**예상 효과**: EM 0.4 → 0.6~0.7

### Phase 2: 모델 용량 증가

**목표**: 더 큰 모델로 표현력 향상

**변경 파일**:
- `config.py`: `d_model=512`, `num_layers=2`

**예상 효과**: EM 0.6 → 0.65~0.75

### Phase 3: Beam Search 도입

**목표**: 추론 시 더 나은 시퀀스 생성

**변경 파일**:
- `model.py`: `generate()` 메서드에 Beam Search 추가
- `config.py`: `beam_size=5` 추가

**예상 효과**: EM 0.65 → 0.7~0.75

### Phase 4: 학습 전략 개선

**목표**: Scheduled Sampling, Learning Rate Schedule 등

**변경 파일**:
- `train.py`: Scheduled Sampling, LR Schedule 추가

**예상 효과**: EM 0.7 → 0.75~0.8

### Phase 5: 데이터 전략 개선

**목표**: Curriculum Learning, 데이터 증강

**변경 파일**:
- `dataloader.py`: 데이터 증강 함수 추가
- `train.py`: Curriculum Learning 로직 추가

**예상 효과**: EM 0.75 → 0.8~0.85

---

## 변경 파일 목록

### 필수 변경 파일

1. **`model.py`** ⭐⭐⭐
   - `TinySeq2Seq` → `AttentionSeq2Seq` 클래스
   - Bahdanau Attention 구현
   - Multi-layer GRU
   - Beam Search 구현

2. **`config.py`** ⭐⭐
   - `ModelConfig`에 새로운 하이퍼파라미터 추가
   - `TrainConfig`에 Beam Search, Scheduled Sampling 설정 추가

3. **`train.py`** ⭐⭐
   - Scheduled Sampling 로직
   - Learning Rate Schedule
   - Beam Search 사용

### 선택적 변경 파일

4. **`dataloader.py`** ⭐
   - 데이터 증강 함수 (교환법칙, 결합법칙 등)
   - Curriculum Learning 지원

5. **`requirements_local.txt`** (로컬 개발용)
   - 추가 라이브러리 없음 (기본 라이브러리만 사용)

---

## 규칙 준수 확인

### ✅ 허용되는 것

- [x] Attention 메커니즘 도입
- [x] Transformer 아키텍처 (선택사항)
- [x] Multi-layer 구조
- [x] Beam Search
- [x] Scheduled Sampling
- [x] Auxiliary Loss (학습 시)
- [x] 데이터 증강 (학습 데이터 생성 시)
- [x] Curriculum Learning

### ❌ 금지되는 것

- [ ] 입력 텍스트 전처리 (문자 제거/추가, 순서 변경)
- [ ] `predict()` 재귀 호출
- [ ] 사후 보정 (Post-processing)
- [ ] 앙상블
- [ ] 명시적 계산 (`eval()`, `int()` 등)

---

## 예상 성능 향상

| 단계 | 개선 사항 | 예상 EM | 누적 효과 |
|------|----------|---------|----------|
| Baseline | 현재 모델 | 0.40 | - |
| Phase 1 | Attention 도입 | 0.65 | +0.25 |
| Phase 2 | 모델 용량 증가 | 0.72 | +0.07 |
| Phase 3 | Beam Search | 0.75 | +0.03 |
| Phase 4 | 학습 전략 개선 | 0.80 | +0.05 |
| Phase 5 | 데이터 전략 개선 | 0.85 | +0.05 |

**최종 목표**: EM 0.85+ (현재 대비 2배 이상 향상)

---

## 다음 단계

1. **Phase 1 구현**: Attention 기반 모델 구현
2. **테스트**: 기본 성능 확인
3. **단계적 개선**: Phase 2~5 순차적 적용
4. **하이퍼파라미터 튜닝**: 각 단계별 최적화
5. **최종 검증**: 다양한 평가 지표 확인

---

**문서 작성일**: 2025년  
**최종 업데이트**: 최적 모델 구현 가이드 작성 완료

