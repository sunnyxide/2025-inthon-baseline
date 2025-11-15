# 모델 한계점 분석 및 개선점 제안

> **작성일**: 2025년  
> **목적**: 현재 TransformerSeq2Seq 모델의 한계점을 분석하고, 실현 가능한 개선 방안을 제시

---

## 📊 현재 모델 구조 요약

### 아키텍처
- **모델 타입**: Transformer 기반 Seq2Seq
- **Encoder**: 6 layers, 2 attention heads, d_model=256
- **Decoder**: 2 layers, 2 attention heads, d_model=256
- **FFN**: dim_feedforward=1024
- **Dropout**: 0.0 (과적합 없음)

### 학습 설정
- **Optimizer**: AdamW (lr=5e-4, weight_decay=0.1)
- **Scheduler**: Cosine Annealing with Warmup (warmup_steps=5000)
- **Batch Size**: 128
- **Gradient Clipping**: 1.0

### 디코딩
- **전략**: Greedy Decoding만 사용
- **Max Length**: 50

---

## 🔴 주요 한계점 분석

### 1. 디코딩 전략의 한계

#### 문제점
- **Greedy Decoding만 사용**: 각 타임스텝에서 가장 확률이 높은 토큰만 선택
- **지역 최적해 문제**: 전체 시퀀스 관점에서 최적이 아닐 수 있음
- **복구 불가능**: 한 번 잘못된 토큰을 생성하면 이후 수정 불가

#### 영향
- 긴 수식에서 오류 누적
- 복잡한 연산 우선순위 처리 실패
- 6자리 이상 출력에서 정확도 저하

#### 코드 위치
```598:657:2025-inthon-baseline/model.py
@torch.no_grad()
def generate(
    self,
    src: torch.Tensor,   # [B, S]
    max_len: int,
    bos_id: int,
    eos_id: int,
    src_pad_id: int,
) -> torch.Tensor:
    # ... greedy decoding only ...
    next_id = torch.argmax(logits, dim=-1)  # [B]
```

---

### 2. RPN 보조 학습 비활성화

#### 문제점
- **lambda_rpn = 0.0**: RPN(Reverse Polish Notation) 보조 디코더가 비활성화됨
- **구현은 있으나 미사용**: `forward_with_rpn()` 메서드와 RPN 디코더가 구현되어 있지만 학습에 활용되지 않음

#### 영향
- 연산 우선순위 학습 기회 상실
- 괄호 처리 능력 제한
- 복잡한 중첩 수식 처리 어려움

#### 코드 위치
```85:85:2025-inthon-baseline/config.py
lambda_rpn: float = 0.0  # Auxiliary loss weight for RPN decoder (0.0 = disabled)
```

---

### 3. Teacher Forcing 고정

#### 문제점
- **teacher_forcing = 1.0 고정**: 학습 시 항상 정답을 사용
- **Scheduled Sampling 없음**: 학습-추론 분포 차이(Exposure Bias) 발생

#### 영향
- 추론 시 오류 누적에 취약
- 첫 번째 토큰 오류 시 전체 시퀀스 실패 가능성 증가

#### 코드 위치
```501:501:2025-inthon-baseline/model.py
teacher_forcing: float = 1.0,  # 현재는 항상 teacher forcing 1.0 (GRU 버전과 동일)
```

---

### 4. Attention Heads 제한

#### 문제점
- **nhead = 2**: Multi-head attention의 head 수가 적음
- **표현력 제한**: 다양한 관계를 동시에 학습하기 어려움

#### 영향
- 복잡한 수식에서 여러 연산자 간 관계 파악 어려움
- 긴 수식에서 장거리 의존성 학습 제한

#### 코드 위치
```18:18:2025-inthon-baseline/config.py
nhead: int = 2  # Attention heads (legacy 최적값)
```

---

### 5. 모델 용량 제한

#### 문제점
- **d_model = 256**: 상대적으로 작은 hidden dimension
- **Encoder/Decoder 비율**: Encoder 6 layers, Decoder 2 layers (비대칭)

#### 영향
- 복잡한 패턴 학습 능력 제한
- 큰 숫자(6자리 이상) 처리 어려움

---

### 6. 데이터 증강 제한

#### 문제점
- **증강 확률 낮음**: Phase별로 15-30%만 증강
- **증강 패턴 제한**: 일부 수학 법칙만 적용

#### 영향
- 일반화 성능 제한
- OOD 데이터 대응 어려움

#### 코드 위치
```44:49:2025-inthon-baseline/dataloader.py
PHASE_AUGMENTATION_PROB = {
    1: 0.15,  # Phase 1: 15% 증강 (기초 단계)
    2: 0.25,  # Phase 2: 25% 증강 (중급)
    3: 0.30,  # Phase 3: 30% 증강 (고급)
    4: 0.20,  # Phase 4: 20% 증강 (큰 숫자는 증강 줄임)
}
```

---

### 7. Positional Encoding 제한

#### 문제점
- **기본 sin/cos PE**: 학습 가능한 positional embedding 없음
- **고정된 패턴**: 모델이 위치 정보를 유연하게 학습하기 어려움

#### 영향
- 긴 시퀀스에서 위치 정보 손실
- 복잡한 괄호 구조 처리 어려움

---

## ✅ 개선점 제안 (우선순위별)

### 🔥 Priority 1: 즉시 적용 가능한 개선

#### 1.1 Beam Search 도입
**목표**: Greedy decoding을 Beam Search로 대체하여 더 나은 시퀀스 생성

**예상 효과**: EM +0.03~0.05

**구현 방법**:
```python
def generate_with_beam_search(
    self,
    src: torch.Tensor,
    max_len: int,
    bos_id: int,
    eos_id: int,
    src_pad_id: int,
    beam_size: int = 5,
) -> torch.Tensor:
    # Beam search 구현
    # 각 타임스텝에서 top-k 후보 유지
    # 최종적으로 가장 높은 점수의 시퀀스 선택
```

**변경 파일**: `model.py` - `TransformerSeq2Seq.generate()` 메서드

---

#### 1.2 RPN 보조 학습 활성화
**목표**: lambda_rpn을 0.1~0.2로 설정하여 연산 우선순위 학습 강화

**예상 효과**: EM +0.02~0.04 (특히 복잡한 수식에서)

**구현 방법**:
```python
# config.py
lambda_rpn: float = 0.15  # 0.0 → 0.15

# train.py에서 이미 구현되어 있음
# use_rpn_head가 True가 되도록 설정
```

**변경 파일**: `config.py` - `TrainConfig.lambda_rpn`

---

#### 1.3 Scheduled Sampling 도입
**목표**: Teacher forcing 비율을 점진적으로 감소시켜 Exposure Bias 완화

**예상 효과**: EM +0.02~0.03

**구현 방법**:
```python
# train.py의 train_loop에서
teacher_forcing_ratio = max(0.5, 1.0 - (step / max_steps) * 0.5)

# forward 호출 시
logits = model(src, tgt_inp, src_pad_id, teacher_forcing=teacher_forcing_ratio)
```

**변경 파일**: 
- `model.py` - `forward()` 메서드에 teacher_forcing 적용
- `train.py` - `train_loop()`에서 scheduled sampling 로직 추가

---

### ⚡ Priority 2: 중기 개선 (하이퍼파라미터 튜닝)

#### 2.1 Attention Heads 증가
**목표**: nhead를 2 → 4 또는 8로 증가

**예상 효과**: EM +0.01~0.02

**주의사항**: d_model이 nhead로 나누어떨어져야 함 (256 → 4 또는 8 가능)

**변경 파일**: `config.py` - `ModelConfig.nhead`

---

#### 2.2 모델 용량 증가
**목표**: d_model을 256 → 384 또는 512로 증가

**예상 효과**: EM +0.02~0.04

**트레이드오프**: 학습 시간 증가, 메모리 사용량 증가

**변경 파일**: `config.py` - `ModelConfig.d_model`

---

#### 2.3 Decoder Layers 증가
**목표**: num_decoder_layers를 2 → 3 또는 4로 증가

**예상 효과**: EM +0.01~0.02

**변경 파일**: `config.py` - `ModelConfig.num_decoder_layers` 또는 depth_profile 사용

---

### 🎯 Priority 3: 장기 개선 (아키텍처 변경)

#### 3.1 Learnable Positional Embedding
**목표**: 고정된 sin/cos PE를 학습 가능한 embedding으로 변경

**예상 효과**: EM +0.01~0.02

**구현 방법**:
```python
# model.py
self.pos_emb_in = nn.Parameter(torch.randn(1, max_len, d_model))
self.pos_emb_out = nn.Parameter(torch.randn(1, max_len, d_model))
```

**변경 파일**: `model.py` - `TransformerSeq2Seq.__init__()`

---

#### 3.2 데이터 증강 확대
**목표**: 증강 확률을 30-50%로 증가, 더 다양한 패턴 추가

**예상 효과**: EM +0.01~0.02 (일반화 성능 향상)

**변경 파일**: `dataloader.py` - `PHASE_AUGMENTATION_PROB` 및 증강 함수 추가

---

#### 3.3 Label Smoothing
**목표**: Cross-entropy loss에 label smoothing 적용

**예상 효과**: EM +0.01 (과적합 완화)

**구현 방법**:
```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=output_tokenizer.pad_id,
    label_smoothing=0.1
)
```

**변경 파일**: `train.py` - `train_loop()` 함수

---

## 📈 예상 성능 향상 시나리오

### 시나리오 A: 보수적 개선 (Priority 1만 적용)
- **Beam Search (beam_size=5)**: +0.03
- **RPN 활성화 (lambda_rpn=0.15)**: +0.02
- **Scheduled Sampling**: +0.02
- **총 예상 향상**: **+0.07** (현재 대비 약 10-15% 향상)

### 시나리오 B: 적극적 개선 (Priority 1 + 2)
- **시나리오 A**: +0.07
- **nhead=4**: +0.01
- **d_model=384**: +0.02
- **Decoder layers=3**: +0.01
- **총 예상 향상**: **+0.11** (현재 대비 약 15-20% 향상)

### 시나리오 C: 최대 개선 (Priority 1 + 2 + 3)
- **시나리오 B**: +0.11
- **Learnable PE**: +0.01
- **데이터 증강 확대**: +0.01
- **Label Smoothing**: +0.01
- **총 예상 향상**: **+0.14** (현재 대비 약 20-25% 향상)

---

## 🛠️ 구현 우선순위 및 단계별 계획

### Phase 1: 즉시 적용 (1-2일)
1. ✅ Beam Search 구현 및 테스트
2. ✅ RPN 활성화 (lambda_rpn=0.15)
3. ✅ Scheduled Sampling 구현

### Phase 2: 하이퍼파라미터 튜닝 (3-5일)
1. ✅ nhead 증가 실험 (2 → 4)
2. ✅ d_model 증가 실험 (256 → 384)
3. ✅ Decoder layers 증가 실험 (2 → 3)

### Phase 3: 아키텍처 개선 (5-7일)
1. ✅ Learnable Positional Embedding
2. ✅ 데이터 증강 확대
3. ✅ Label Smoothing 적용

---

## ⚠️ 주의사항

### 1. 메모리 및 계산 비용
- Beam Search: 메모리 사용량이 beam_size배 증가
- 모델 용량 증가: 학습 시간 및 메모리 사용량 증가
- RPN 활성화: 추가 디코더로 인한 계산 비용 증가

### 2. 규칙 준수
- ✅ 모든 개선사항은 대회 규칙을 준수
- ✅ 입력 텍스트 전처리 없음
- ✅ predict() 재귀 호출 없음
- ✅ 명시적 계산 없음

### 3. 체크포인트 호환성
- 기존 `best_model.pt`와의 호환성 고려
- `strict=False`로 로딩하여 새로운 파라미터는 랜덤 초기화

---

## 📝 결론

현재 모델은 Transformer 기반으로 잘 구현되어 있으나, **디코딩 전략**, **보조 학습 비활성화**, **Teacher Forcing 고정** 등의 한계가 있습니다.

**즉시 적용 가능한 개선사항(Priority 1)**만으로도 **약 7-10%의 성능 향상**을 기대할 수 있으며, 이를 통해 복잡한 수식 처리 능력이 크게 개선될 것입니다.

가장 큰 효과를 기대할 수 있는 개선사항:
1. **Beam Search 도입** (가장 큰 효과)
2. **RPN 보조 학습 활성화** (복잡한 수식에서 효과)
3. **Scheduled Sampling** (일반화 성능 향상)

---

**다음 단계**: Priority 1 개선사항부터 순차적으로 구현 및 실험 진행

