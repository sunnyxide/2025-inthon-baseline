# 모델 분석 및 한계점 리포트

## 1. 모델 구조 분석

### 1.1 아키텍처 개요
현재 모델은 **TinySeq2Seq**이라는 매우 단순한 GRU 기반 Seq2Seq 모델입니다.

```
입력 수식 (예: "3335+472")
    ↓
[CharTokenizer] → 문자 단위 토큰화
    ↓
[Embedding] → d_model=256 차원 벡터
    ↓
[Encoder GRU] → 마지막 hidden state만 사용 (bottleneck!)
    ↓
[Decoder GRU] → hidden state로부터 출력 생성
    ↓
[Linear Projection] → vocab 크기 logits
    ↓
[Greedy Decoding] → 가장 높은 확률의 토큰 선택
    ↓
출력 숫자 (예: "3807")
```

### 1.2 주요 구성 요소

#### 인코더 (Encoder)
- **구조**: 단일 GRU 레이어 (d_model=256)
- **입력**: 수식 문자열의 문자 단위 임베딩
- **출력**: 마지막 타임스텝의 hidden state만 사용
- **문제점**: 
  - 입력 시퀀스의 모든 정보를 단일 벡터(256차원)에 압축해야 함
  - 긴 수식의 경우 정보 손실 발생

#### 디코더 (Decoder)
- **구조**: 단일 GRU 레이어 (d_model=256)
- **입력**: 인코더의 마지막 hidden state + 이전 출력 토큰
- **출력**: 각 타임스텝의 vocab 크기 logits
- **문제점**:
  - Attention 메커니즘 없음
  - 인코더의 전체 출력을 활용하지 못함

#### 토크나이저
- **입력**: `0123456789+-*/()` (18개 문자 + 특수 토큰)
- **출력**: `0123456789` (10개 숫자 + 특수 토큰)
- **방식**: 문자 단위 토크나이징

### 1.3 학습 설정

```python
# 모델 설정
d_model = 256  # Hidden dimension (작은 모델)

# 학습 설정
learning_rate = 2e-3
batch_size = 128
num_epochs = 50
max_gen_len = 24

# 데이터 설정
train: max_depth=3, num_digits=(1,3)  # 상대적으로 쉬운 데이터
val:   max_depth=4, num_digits=(1,5)  # 더 어려운 데이터
```

## 2. 학습 결과 분석

### 2.1 성능 지표 추이

| Step | EM | TES | 특징 |
|------|-----|-----|------|
| 200 | 0.305 | 0.507 | 초기 학습 |
| 400 | 0.352 | 0.566 | 빠른 개선 |
| 2400 | 0.359 | 0.579 | 정체 시작 |
| 4000 | 0.367 | 0.595 | 완만한 개선 |
| 10000 | 0.383 | 0.632 | 최고 성능 근접 |
| 20000 | 0.406 | 0.662 | 최고 EM 달성 |
| 34400 | 0.406 | 0.672 | 정체 (과적합 의심) |

### 2.2 예측 패턴 분석

#### 성공 사례
- **단순 숫자**: `28 → 28` ✅
  - 단일 숫자는 거의 항상 정확

#### 실패 사례
1. **단순 숫자 복사 실패**: `97725 → 975`
   - 긴 숫자의 뒷부분 손실
   - 디코더가 EOS를 너무 빨리 생성

2. **덧셈 실패**: `3335+472 → 807` (정답: 3807)
   - 수식 구조 이해 부족
   - 연산자 처리 실패

3. **복잡한 수식 실패**: `26803+9*34741 → 4246` (정답: 339472)
   - 연산자 우선순위 미반영
   - 큰 숫자 처리 어려움

4. **나눗셈 포함 수식**: `278//698+90538*75 → 69030` (정답: 6790350)
   - 여러 연산자 조합 실패
   - 큰 결과값 생성 어려움

### 2.3 주요 문제점

1. **정확도 정체**: EM이 0.4 수준에서 정체
   - 더 이상 개선되지 않음
   - 모델 용량 부족으로 인한 한계

2. **TES는 개선되지만 EM은 정체**
   - 부분적으로 맞는 답을 생성 (예: 97725 → 975)
   - 완전히 정확한 답은 생성하지 못함

3. **복잡한 수식 처리 실패**
   - 연산자 우선순위 미반영
   - 큰 숫자 처리 어려움

## 3. 모델의 한계점

### 3.1 아키텍처 한계

#### 1) Attention 메커니즘 부재 ⚠️ **가장 큰 문제**
```python
# 현재: 인코더의 마지막 hidden state만 사용
enc_out, h = self.encoder(x)  # h: [1, B, d_model]
dec_out, _ = self.decoder(y, h)  # h만 사용, enc_out은 버려짐
```

**문제점**:
- 입력 시퀀스의 모든 정보를 단일 벡터에 압축해야 함
- 긴 수식의 경우 정보 손실 발생
- 디코더가 입력의 특정 부분에 집중할 수 없음

**해결책**:
- Attention 메커니즘 도입 (Bahdanau, Luong, 또는 Multi-Head Attention)
- Transformer 아키텍처 고려

#### 2) 단일 레이어 구조
```python
self.encoder = nn.GRU(d_model, d_model, batch_first=True)  # 단일 레이어
self.decoder = nn.GRU(d_model, d_model, batch_first=True)  # 단일 레이어
```

**문제점**:
- 표현력 부족
- 복잡한 패턴 학습 어려움

**해결책**:
- 다층 GRU/LSTM 사용 (num_layers=2~4)
- Residual connection 추가

#### 3) 모델 용량 부족
```python
d_model = 256  # 작은 hidden dimension
```

**문제점**:
- 복잡한 수식 처리에 필요한 정보 저장 공간 부족

**해결책**:
- d_model 증가 (512, 768, 1024 등)
- 더 큰 모델 사용

### 3.2 학습 전략 한계

#### 1) Greedy Decoding만 사용
```python
next_id = torch.argmax(logits, dim=-1)  # Greedy decoding
```

**문제점**:
- 각 타임스텝에서 가장 확률이 높은 토큰만 선택
- 전체 시퀀스 관점에서 최적이 아닐 수 있음

**해결책**:
- Beam Search 도입 (beam_size=3~5)
- Top-k/Top-p sampling

#### 2) Teacher Forcing만 사용
```python
# 학습 시 항상 정답을 입력으로 사용
logits = model(src, target_input, ...)  # teacher_forcing=1.0
```

**문제점**:
- 학습과 추론 시 입력 분포 차이 (exposure bias)
- 오류 전파 문제

**해결책**:
- Scheduled Sampling 도입
- Teacher Forcing 비율 점진적 감소

#### 3) 학습 데이터와 검증 데이터 난이도 차이
```python
train: max_depth=3, num_digits=(1,3)  # 쉬운 데이터
val:   max_depth=4, num_digits=(1,5)  # 어려운 데이터
```

**문제점**:
- 검증 데이터가 더 어려워 일반화 성능 저하
- 학습 데이터에 과적합

**해결책**:
- 학습 데이터 난이도 증가
- Curriculum Learning 적용

### 3.3 수식 처리 한계

#### 1) 연산자 우선순위 미반영
- 현재 모델은 수식을 단순 문자 시퀀스로 처리
- `*`, `//`가 `+`, `-`보다 우선순위가 높다는 것을 학습하지 못함

**해결책**:
- 수식 전처리 (후위 표기법 변환 등)
- 구조화된 표현 학습

#### 2) 큰 숫자 처리 어려움
- 긴 숫자 시퀀스 생성 시 뒷부분 손실
- 디코더가 EOS를 너무 빨리 생성

**해결책**:
- 더 긴 max_gen_len 설정
- 숫자 생성 전용 디코딩 전략

#### 3) 음수 처리 부재
- 현재 데이터는 음수가 없지만, 실제로는 음수도 처리해야 할 수 있음

## 4. 개선 방안

### 4.1 즉시 적용 가능한 개선

1. **모델 용량 증가**
   ```python
   d_model = 512  # 256 → 512
   num_layers = 2  # 단일 레이어 → 다층
   ```

2. **Beam Search 도입**
   ```python
   # greedy 대신 beam search 사용
   beam_size = 5
   ```

3. **학습 데이터 난이도 증가**
   ```python
   train: max_depth=4, num_digits=(1,5)  # 검증과 동일한 난이도
   ```

4. **Learning Rate 조정**
   ```python
   lr = 1e-3  # 2e-3 → 1e-3 (더 안정적인 학습)
   ```

### 4.2 중기 개선 방안

1. **Attention 메커니즘 도입**
   - Bahdanau Attention 또는 Luong Attention
   - 입력 시퀀스의 모든 타임스텝 활용

2. **Transformer 아키텍처**
   - Self-Attention과 Cross-Attention 활용
   - 더 강력한 표현력

3. **Scheduled Sampling**
   - Teacher Forcing 비율 점진적 감소
   - 학습-추론 분포 차이 완화

### 4.3 장기 개선 방안

1. **구조화된 표현 학습**
   - 수식을 트리 구조로 표현
   - 연산자 우선순위 명시적 학습

2. **데이터 증강**
   - 더 다양한 수식 패턴 생성
   - 큰 숫자 처리 연습 데이터 증가

3. **앙상블**
   - 여러 모델의 예측 결합
   - 성능 향상

## 5. 결론

현재 모델은 **기본적인 Seq2Seq 구조**로, 단순한 수식은 처리할 수 있지만 복잡한 수식에서는 한계가 명확합니다.

**핵심 문제점**:
1. ⚠️ **Attention 메커니즘 부재** - 가장 큰 문제
2. 단일 레이어 구조로 인한 표현력 부족
3. 모델 용량 부족 (d_model=256)
4. Greedy Decoding의 한계

**예상 개선 효과**:
- Attention 도입: EM 0.4 → 0.6~0.7 예상
- 모델 용량 증가: EM 0.4 → 0.5~0.6 예상
- Beam Search: EM 0.4 → 0.45~0.5 예상
- 종합 개선: EM 0.4 → 0.7~0.8 예상

**우선순위**:
1. **Attention 메커니즘 도입** (최우선)
2. 모델 용량 증가
3. Beam Search 도입
4. 학습 데이터 개선

