# A100 GPU 최적화 가이드

## 개요

이 문서는 A100 GPU를 활용한 대형 모델 학습을 위한 설정 변경 사항을 설명합니다.

## 주요 변경 사항

### 1. 모델 아키텍처 확장 (ModelConfig)

| 파라미터 | 이전 값 (W&B sweep 최적값) | A100 최적화 값 | 변경 이유 |
|---------|------------------------|--------------|---------|
| `d_model` | 256 | 512 | Hidden dimension 2배 증가 - 더 풍부한 표현력 |
| `nhead` | 2 | 8 | Attention head 4배 증가 - 더 다양한 패턴 학습 |
| `num_encoder_layers` | 6 | 8 | Encoder layer 증가 - 입력 이해도 향상 |
| `num_decoder_layers` | 2 | 4 | Decoder layer 2배 증가 - 출력 생성 품질 향상 |
| `dim_feedforward` | 1024 | 2048 | FFN dimension 2배 증가 - 비선형 변환 능력 강화 |
| `dropout` | 0.0 | 0.0 | 유지 (W&B sweep 최적값) |

**모델 파라미터 수 비교:**
- 이전 모델 (d_model=256): ~3M parameters
- A100 최적화 모델 (d_model=512): ~12M parameters
- **약 4배 증가**

### 2. 학습 설정 조정 (TrainConfig)

| 파라미터 | 이전 값 | A100 최적화 값 | 변경 이유 |
|---------|--------|--------------|---------|
| `batch_size` | 128 | 512 | A100 메모리 활용 최적화 (4배 증가) |
| `lr` | 2e-4 (fine-tuning) | 3e-4 | 대형 모델 학습 안정화 |
| `warmup_steps` | 5,000 | 8,000 | 대형 데이터셋에 맞춘 warmup |
| `valid_every` | 200 | 500 | 대형 데이터셋에서 검증 빈도 조정 |
| `num_epochs` | 20 | 30 | 대형 모델 수렴을 위한 epoch 증가 |
| `show_valid_samples` | 5 | 10 | 더 다양한 검증 샘플 확인 |
| `early_stopping_patience` | 5 | 8 | 대형 모델의 긴 수렴 시간 고려 |

### 3. 데이터셋 확장

| 구분 | 이전 설정 | A100 최적화 설정 | 변경 이유 |
|-----|---------|----------------|---------|
| **Training** |
| `num_samples` | 200,000 | 800,000 | 대형 모델 학습용 데이터 4배 증가 |
| `phase` | 2 (2-3자리) | 3 (3-4자리) | 더 복잡한 문제 학습 |
| **Validation** |
| `num_samples` | 1,000 | 2,000 | 검증 샘플 2배 증가 |
| `phase` | 3 (3-4자리) | 4 (4-5자리) | 더 어려운 검증 데이터 |

## 체크포인트 호환성

### ⚠️ 중요: 모델 구조 변경으로 인한 비호환성

**이전 best_model.pt (d_model=256)와 새 모델 (d_model=512)은 호환되지 않습니다.**

#### 호환성 체크 메커니즘

`train.py`의 체크포인트 로드 시 자동으로 설정을 비교합니다:

```python
# 체크포인트 설정과 현재 모델 설정 비교
if config_match:
    # 설정이 일치하면 체크포인트에서 로드
    model.load_state_dict(checkpoint["model_state"])
else:
    # 설정이 다르면 새로 학습 시작
    print("🚀 Starting training from scratch with A100-optimized model.")
```

#### 옵션 1: 새로운 학습 시작 (권장)

A100 GPU의 성능을 최대한 활용하려면 **새로운 학습을 시작**하는 것을 권장합니다:

```bash
python train.py
```

- ✅ A100 GPU 성능 완전 활용
- ✅ 대형 모델 + 대형 데이터셋으로 성능 향상 가능
- ✅ 800k 샘플로 더 다양한 패턴 학습

#### 옵션 2: 이전 체크포인트 사용

이전 `best_model.pt`를 계속 사용하려면 `config.py`를 이전 설정으로 되돌려야 합니다:

```python
# config.py에서 이전 설정으로 변경
class ModelConfig:
    d_model: int = 256
    nhead: int = 2
    num_encoder_layers: int = 6
    num_decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.0
```

## 예상 성능 향상

### 모델 크기 증가 효과

| 측면 | 예상 효과 |
|-----|---------|
| **표현력** | 4배 더 많은 파라미터로 복잡한 패턴 학습 가능 |
| **일반화** | 800k 샘플로 다양한 케이스 학습 → OOD 성능 향상 |
| **수렴 속도** | Batch size 4배 증가 → 같은 epoch에서 4배 빠른 수렴 |
| **메모리 사용** | A100 80GB 메모리를 효율적으로 활용 |

### 학습 시간 예측

```
설정 기준:
- Training samples: 800,000
- Batch size: 512
- Epochs: 30

계산:
- Steps per epoch = 800,000 / 512 ≈ 1,563 steps
- Total steps = 1,563 × 30 ≈ 46,890 steps
- Validation = 46,890 / 500 ≈ 94회

예상 소요 시간 (A100 기준):
- Step당 ~0.5초 가정
- Total: 46,890 × 0.5s ≈ 23,445초 ≈ 6.5시간
```

## 실행 방법

### 1. 환경 확인

```bash
# GPU 확인
python check_gpu.py

# 출력 예시:
# CUDA available: True
# Device: NVIDIA A100-SXM4-80GB
# Memory: 81,050 MB
```

### 2. 학습 시작

```bash
cd /Users/sunny/datathon/2025-inthon-baseline
python train.py
```

### 3. WandB 모니터링

학습 중 다음 지표들을 모니터링하세요:

- **train/loss**: 학습 손실 (감소 추세)
- **train/lr**: 학습률 (warmup → cosine decay)
- **valid/EM**: Exact Match (주요 지표)
- **valid/TES**: Token Exact Score

## 메모리 사용량 예측

### A100 80GB 메모리 사용 예측

```
모델 파라미터: ~12M × 4 bytes (fp32) = 48MB
Optimizer 상태: 48MB × 2 (AdamW) = 96MB
Batch 데이터 (512 batch):
  - Input: 512 × 50 × 4 bytes ≈ 100KB
  - Hidden states: 512 × 50 × 512 × 4 bytes ≈ 50MB
  - Gradients: ~48MB

총 예상 메모리: ~250MB (매우 안전)
```

**A100 80GB 메모리를 고려하면 batch_size를 더 늘릴 수도 있습니다!**

### Batch Size 추가 확장 가능성

현재 batch_size=512로 설정했지만, A100의 메모리가 충분하면 더 늘릴 수 있습니다:

| Batch Size | 예상 메모리 | 학습 속도 | 권장 사용 |
|-----------|-----------|---------|---------|
| 512 | ~250MB | 기준 | ✅ 현재 설정 (안정적) |
| 1024 | ~500MB | 2배 빠름 | ✅ 가능 (메모리 여유) |
| 2048 | ~1GB | 4배 빠름 | ⚠️ 실험 필요 |

## 트러블슈팅

### OOM (Out of Memory) 발생 시

1. **Batch size 줄이기**
   ```python
   # config.py
   batch_size: int = 256  # 512 → 256
   ```

2. **모델 크기 줄이기**
   ```python
   # config.py
   d_model: int = 384  # 512 → 384
   ```

3. **Mixed precision 학습 (향후 적용 가능)**
   ```python
   # train.py에 추가
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### 학습이 불안정한 경우

1. **Learning rate 낮추기**
   ```python
   lr: float = 1e-4  # 3e-4 → 1e-4
   ```

2. **Warmup steps 늘리기**
   ```python
   warmup_steps: int = 10000  # 8000 → 10000
   ```

3. **Gradient clipping 강화**
   ```python
   grad_clip: float = 0.5  # 1.0 → 0.5
   ```

## 개발자 로그

**변경 일시**: 2025-11-15  
**브랜치**: `feature/scale-for-a100-gpu`  
**작업자**: AI Assistant

### 변경 파일
- `config.py`: ModelConfig, TrainConfig A100 최적화
- `train.py`: 데이터셋 확장, 체크포인트 호환성 체크 추가

### 주요 설계 결정
1. **모델 크기 4배 증가**: A100 GPU의 계산 능력 활용
2. **데이터셋 4배 증가**: 대형 모델의 과적합 방지
3. **Batch size 4배 증가**: A100 메모리 효율적 활용
4. **호환성 체크 추가**: 이전 체크포인트와 자동 비교

### 테스트 필요 사항
- [ ] A100 GPU에서 메모리 사용량 확인
- [ ] 학습 안정성 확인 (첫 1000 steps)
- [ ] Valid EM 성능 추이 모니터링
- [ ] Batch size 추가 증가 실험 (512 → 1024)

## Git Commit 메시지

```
feat: Scale model and dataset for A100 GPU optimization

- Model architecture scaling (4x parameters)
  * d_model: 256 → 512
  * nhead: 2 → 8
  * encoder_layers: 6 → 8, decoder_layers: 2 → 4
  * dim_feedforward: 1024 → 2048

- Dataset expansion (4x samples)
  * Training: 200k → 800k samples (phase 2 → 3)
  * Validation: 1k → 2k samples (phase 3 → 4)

- Training configuration optimization
  * batch_size: 128 → 512 (A100 memory utilization)
  * lr: 2e-4 → 3e-4 (large model stabilization)
  * warmup_steps: 5k → 8k, epochs: 20 → 30
  * valid_every: 200 → 500 (large dataset)

- Checkpoint compatibility check
  * Auto-detect config mismatch with old checkpoints
  * Gracefully handle incompatible models
  * Provide clear guidance for users

Expected improvements:
- 4x model capacity for complex pattern learning
- Better OOD generalization with larger dataset
- Efficient A100 GPU utilization

Breaking changes:
- Old best_model.pt (d_model=256) is not compatible
- New training from scratch recommended for full performance
```

## 참고 문헌

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 아키텍처
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 모델 스케일링 법칙
- [NVIDIA A100 Datasheet](https://www.nvidia.com/en-us/data-center/a100/) - A100 GPU 사양

