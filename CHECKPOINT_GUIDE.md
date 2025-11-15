# 체크포인트 로드 및 학습 재개 가이드

이 가이드는 체크포인트를 검증하고 학습을 재개하는 방법을 설명합니다.

---

## 📋 목차

1. [체크포인트 검증하기](#1-체크포인트-검증하기)
2. [학습 재개하기](#2-학습-재개하기)
3. [Wandb 최적 모델 다운로드](#3-wandb-최적-모델-다운로드)
4. [문제 해결](#4-문제-해결)

---

## 1. 체크포인트 검증하기

학습을 재개하기 전에 **반드시** 체크포인트를 검증하세요!

### 방법 1: 검증 스크립트 사용 (권장)

```bash
# 로컬 또는 Colab에서
python verify_checkpoint.py

# 또는 특정 파일 검증
python verify_checkpoint.py my_checkpoint.pt

# 추론 테스트 제외
python verify_checkpoint.py --no-inference
```

**출력 예시:**

```
======================================================================
🔍 CHECKPOINT VERIFICATION
======================================================================
✅ Checkpoint found: best_model.pt
📦 Size: 45.23 MB

✅ Checkpoint loaded successfully

📝 Checkpoint Contents:
  ✅ model_state       : 142 parameters
  ✅ optim_state       : optimizer state saved
  ✅ step              : 15000
  ✅ model_config      : configuration saved
  ✅ train_config      : configuration saved

🔧 Model Configuration Comparison:
Parameter                  Current      Saved     Status
------------------------------------------------------------
d_model                        256        256  ✅ MATCH
nhead                            2          2  ✅ MATCH
num_encoder_layers               6          6  ✅ MATCH
num_decoder_layers               2          2  ✅ MATCH
dim_feedforward               1024       1024  ✅ MATCH
dropout                        0.0        0.0  ✅ MATCH

✅ ✅ ✅ ALL CONFIGURATIONS MATCH! ✅ ✅ ✅

🔄 Testing Weight Loading...
✅ Weights loaded successfully
✅ Weights successfully updated from checkpoint

📊 Weight Sample (encoder.layers[0].linear1.weight[0, :3]):
   [0.0234, -0.0156, 0.0423]

🧪 Running Inference Test...

Expression      Predicted    Expected   Status
-------------------------------------------------------
2+3                     5           5  ✅ PASS
10-5                    5           5  ✅ PASS
3*4                    12          12  ✅ PASS
(2+3)*4                20          20  ✅ PASS

📊 Inference Test Results: 4/4 passed
✅ All inference tests passed!

======================================================================
✅ ✅ ✅ VERIFICATION COMPLETE ✅ ✅ ✅
======================================================================

📋 Summary:
  • Checkpoint file:   best_model.pt
  • File size:         45.23 MB
  • Configuration:     ✅ Match
  • Weights loaded:    ✅ Success
  • Inference test:    4/4 passed

🚀 Ready to resume training! Set resume_from_checkpoint=True in train.py
======================================================================
```

### 방법 2: 수동 검증 (Python 코드)

```python
import torch
import os

# 체크포인트 로드
checkpoint_path = "best_model.pt"
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint keys:", list(ckpt.keys()))
    
    if "model_config" in ckpt:
        print("\nModel Config:")
        for k, v in ckpt["model_config"].items():
            print(f"  {k}: {v}")
    
    if "step" in ckpt:
        print(f"\nSaved at step: {ckpt['step']}")
```

---

## 2. 학습 재개하기

### Step 1: train.py 수정

`train.py`의 `main()` 함수에서 다음 줄을 수정:

```python
# 이 줄 찾기 (약 755번째 줄)
resume_from_checkpoint = False  # ← 이 줄을 수정

# False를 True로 변경
resume_from_checkpoint = True
```

### Step 2: 체크포인트 파일 위치 확인

기본적으로 `best_model.pt`를 찾습니다. 다른 파일을 사용하려면:

```python
resume_checkpoint = "my_checkpoint.pt"  # 파일 이름 변경
resume_from_checkpoint = True
```

### Step 3: 학습 시작

```bash
# 로컬
python train.py

# Colab
!python train.py
```

**재개 시 출력 예시:**

```
======================================================================
🔄 Resuming from checkpoint: best_model.pt
======================================================================

📋 Checkpoint Configuration:
  ✅ d_model           : current=   256, saved=   256
  ✅ nhead             : current=     2, saved=     2
  ✅ num_encoder_layers: current=     6, saved=     6
  ✅ num_decoder_layers: current=     2, saved=     2
  ✅ dim_feedforward   : current=  1024, saved=  1024
  ✅ dropout           : current=   0.0, saved=   0.0

✅ Configuration matches!

✅ Model weights loaded successfully
📊 Resuming from step: 15000
======================================================================
```

---

## 3. Wandb 최적 모델 다운로드

### Step 1: Wandb에서 최적 Run 찾기

1. Wandb 프로젝트 페이지 접속
2. **Runs** 탭에서 정렬: `valid/EM` 기준 내림차순
3. 최고 EM을 가진 Run 클릭
4. **Files** 탭 이동
5. `best_model.pt` 다운로드

### Step 2: Colab에 업로드

```python
from google.colab import files

# 파일 업로드
uploaded = files.upload()

# 업로드된 파일 확인
!ls -lh best_model.pt
```

### Step 3: 검증 후 학습 재개

```python
# 1. 검증
!python verify_checkpoint.py best_model.pt

# 2. train.py 수정 (위 Step 1 참고)

# 3. 학습 시작
!python train.py
```

---

## 4. 문제 해결

### ❌ 문제: Configuration Mismatch

**증상:**
```
⚠️  d_model           : current=   256, saved=   512
```

**해결 방법:**

1. **Option A: train.py를 체크포인트에 맞추기** (권장)
   ```python
   # train.py의 model_config 수정
   model_config = ModelConfig(
       d_model=512,  # 체크포인트 값으로 변경
       nhead=4,
       num_encoder_layers=4,
       num_decoder_layers=4,
       dim_feedforward=2048,
       dropout=0.1,
   )
   ```

2. **Option B: 처음부터 다시 학습**
   ```python
   resume_from_checkpoint = False
   ```

---

### ❌ 문제: Weights 로드 실패

**증상:**
```
RuntimeError: Error(s) in loading state_dict for TransformerSeq2Seq:
        size mismatch for encoder.layers.0.linear1.weight: 
        copying a param with shape torch.Size([1024, 256]) from checkpoint, 
        the shape in current model is torch.Size([512, 256]).
```

**원인:** 모델 구조가 체크포인트와 다름

**해결 방법:**
1. `verify_checkpoint.py` 실행하여 설정 확인
2. train.py의 `model_config`를 체크포인트와 일치시키기

---

### ❌ 문제: Checkpoint 파일을 찾을 수 없음

**증상:**
```
⚠️  Checkpoint file not found: best_model.pt
Starting training from scratch...
```

**해결 방법:**

```bash
# 파일 위치 확인
ls -l best_model.pt

# 파일이 다른 위치에 있다면
resume_checkpoint = "/content/my_folder/best_model.pt"
```

---

### ❌ 문제: 추론 테스트 실패

**증상:**
```
Expression      Predicted    Expected   Status
-------------------------------------------------------
2+3               ERROR            5  ❌ FAIL
```

**원인:** 
- 모델이 충분히 학습되지 않음
- Weight가 제대로 로드되지 않음

**해결 방법:**
1. 더 학습된 체크포인트 사용
2. 체크포인트 무결성 확인
3. 처음부터 다시 학습

---

## 5. 체크포인트 구조

### 저장되는 내용

`best_model.pt`에 저장되는 내용:

```python
{
    "model_state": OrderedDict(...),      # 모델 weights
    "optim_state": {...},                 # Optimizer 상태
    "step": 15000,                        # 학습 step
    "model_config": {                     # 모델 설정
        "d_model": 256,
        "nhead": 2,
        ...
    },
    "train_config": {                     # 학습 설정
        "lr": 0.0005,
        "num_epochs": 20,
        ...
    },
    "tokenizer_config": {                 # 토크나이저 설정
        "input_chars": "...",
        "output_chars": "...",
        ...
    }
}
```

### 체크포인트 크기 참고

- **작은 모델** (d_model=128): ~10-20 MB
- **중간 모델** (d_model=256): ~40-60 MB
- **큰 모델** (d_model=512): ~150-200 MB

---

## 6. 빠른 체크리스트

학습 재개 전 체크리스트:

- [ ] `verify_checkpoint.py` 실행
- [ ] ✅ All configurations match 확인
- [ ] ✅ Weights loaded successfully 확인
- [ ] 추론 테스트 통과 (선택)
- [ ] `train.py`에서 `resume_from_checkpoint = True` 설정
- [ ] 학습 시작!

---

## 7. 추가 팁

### Tip 1: 정기적 체크포인트 저장

현재는 Best EM일 때만 저장됩니다. 정기적 저장을 원하면:

```python
# train_loop에서 추가
if step % 5000 == 0:  # 5000 step마다
    torch.save({
        "model_state": model.state_dict(),
        "step": step,
    }, f"checkpoint_step_{step}.pt")
```

### Tip 2: 여러 체크포인트 비교

```bash
# 여러 파일 검증
for file in *.pt; do
    echo "Checking $file..."
    python verify_checkpoint.py "$file" --no-inference
    echo ""
done
```

### Tip 3: GPU 메모리 부족 시

```python
# CPU에서 로드 후 GPU로 이동
checkpoint = torch.load("best_model.pt", map_location='cpu')
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
```

---

## 문의사항

문제가 발생하면:
1. `verify_checkpoint.py` 실행 결과 확인
2. 에러 메시지 전문 복사
3. 체크포인트 구조 확인: `torch.load("best_model.pt").keys()`

---

**마지막 업데이트:** 2025-11-15

