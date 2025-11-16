# Colab에서 RPN 키 오류 해결하기

## 🚀 빠른 해결 방법

### Step 1: Clean Checkpoint 생성

Colab에서 다음 코드 셀을 실행하세요:

```python
# Clean checkpoint 생성 (한 번만 실행)
import torch
from pathlib import Path

input_path = "best_model.pt"
output_path = "best_model_clean.pt"

print(f"📂 Loading checkpoint from: {input_path}")

# 체크포인트 로드
checkpoint = torch.load(input_path, map_location="cpu")

# model_state 추출
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model_state = checkpoint["model_state"]
elif isinstance(checkpoint, dict):
    model_state = checkpoint
else:
    model_state = checkpoint

print(f"   Original keys: {len(model_state)}")

# RPN 키 제거
rpn_keys = [k for k in model_state.keys() if k.startswith("rpn_")]
print(f"   Found {len(rpn_keys)} RPN keys")

clean_state = {
    k: v for k, v in model_state.items()
    if not k.startswith("rpn_")
}

print(f"   Clean keys: {len(clean_state)}")
print(f"   Removed: {len(model_state) - len(clean_state)} keys")

# Clean checkpoint 생성
clean_checkpoint = {
    "model_state": clean_state,
}

# Config 보존
if "tokenizer_config" in checkpoint:
    clean_checkpoint["tokenizer_config"] = checkpoint["tokenizer_config"]
    print(f"   ✅ Kept tokenizer_config")

if "model_config" in checkpoint:
    model_config = checkpoint["model_config"].copy()
    if "rpn_vocab" in model_config:
        model_config.pop("rpn_vocab")
        print(f"   ✅ Removed rpn_vocab from model_config")
    clean_checkpoint["model_config"] = model_config
    print(f"   ✅ Kept model_config")

# 저장
print(f"\n💾 Saving clean checkpoint to: {output_path}")
torch.save(clean_checkpoint, output_path)

# 파일 크기 확인
output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
print(f"   ✅ Clean checkpoint saved ({output_size:.2f} MB)")
print(f"\n🎉 Clean checkpoint created successfully!")
print(f"   Now you can use Model() class without RPN key errors")
```

### Step 2: Model 클래스 수정 (최신 코드 반영)

Colab의 `model.py` 파일에서 `Model.__init__` 메서드를 다음으로 교체하세요:

```python
def __init__(self) -> None:
    """
    모델 초기화
    
    체크포인트를 로드하고 모델을 평가 모드로 설정합니다.
    모든 초기화는 이 메서드에서 완료되어야 합니다.
    """
    super().__init__()
    
    # 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 체크포인트 경로 (반드시 상대 경로 사용)
    # Clean checkpoint 우선 사용
    import os
    CKPT_PATH = "best_model_clean.pt" if os.path.exists("best_model_clean.pt") else "best_model.pt"
    
    # 체크포인트 로드
    checkpoint = torch.load(CKPT_PATH, map_location=self.device)
    
    # 토크나이저 설정 로드
    tokenizer_config_dict = checkpoint.get("tokenizer_config")
    if tokenizer_config_dict is None:
        raise ValueError(f"체크포인트에 'tokenizer_config'가 없습니다.")
    
    # 토크나이저 초기화
    input_chars = tokenizer_config_dict.get("input_chars", INPUT_CHARS)
    output_chars = tokenizer_config_dict.get("output_chars", OUTPUT_CHARS)
    add_special = tokenizer_config_dict.get("add_special", True)
    
    self.input_tokenizer = CharTokenizer(
        input_chars if input_chars is not None else INPUT_CHARS,
        add_special=add_special,
    )
    self.output_tokenizer = CharTokenizer(
        output_chars if output_chars is not None else OUTPUT_CHARS,
        add_special=add_special,
    )
    
    # 모델 설정 로드
    model_config_dict = checkpoint.get("model_config")
    if model_config_dict is None:
        raise ValueError(f"체크포인트에 'model_config'가 없습니다.")
    
    # rpn_vocab 제거
    model_config_dict = model_config_dict.copy()
    if "rpn_vocab" in model_config_dict:
        model_config_dict.pop("rpn_vocab")
    
    # Transformer 모델 인스턴스 생성
    self.model = TransformerSeq2Seq(
        in_vocab=self.input_tokenizer.vocab_size,
        out_vocab=self.output_tokenizer.vocab_size,
        rpn_vocab=None,  # 제출용: RPN decoder 사용 안 함
        **model_config_dict,
    ).to(self.device)
    
    # 모델 가중치 로드
    model_state = checkpoint.get("model_state", checkpoint)
    if not isinstance(model_state, dict):
        raise ValueError(f"model_state must be a dict, got {type(model_state)}")
    
    # Always filter RPN keys (defensive approach)
    original_keys = len(model_state)
    rpn_keys = [k for k in model_state.keys() if k.startswith("rpn_")]
    
    if rpn_keys:
        # Filter RPN keys
        print(f"⚠️ Warning: Found {len(rpn_keys)} RPN keys in checkpoint, filtering them...")
        model_state = {k: v for k, v in model_state.items() if not k.startswith("rpn_")}
        filtered_keys = len(model_state)
        print(f"   Filtered: {original_keys} -> {filtered_keys} keys")
    
    # Always use strict=False when filtering (safe approach)
    missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
    
    if missing_keys:
        print(f"⚠️ Warning: {len(missing_keys)} missing keys (using random init)")
    if unexpected_keys:
        print(f"⚠️ Warning: {len(unexpected_keys)} unexpected keys (ignored)")
    
    # 최대 생성 길이 설정
    self.max_len = 50
    
    # 평가 모드로 설정
    self.model.eval()
```

### Step 3: 평가 실행

이제 `evaluate_arithmetic.py`를 실행하면 정상 작동합니다:

```python
!python evaluate_arithmetic.py
```

---

## 🔍 문제 해결

### Q: Clean checkpoint 생성 후에도 오류 발생

**A**: `model.py`의 `Model.__init__` 메서드가 최신 코드로 업데이트되었는지 확인하세요.

### Q: 파일이 너무 커서 업로드가 느림

**A**: Clean checkpoint는 원본보다 작습니다 (RPN 파라미터 제거). 그래도 큰 경우 Google Drive에 업로드하고 마운트하세요.

### Q: Git으로 최신 코드 받기

**A**: Colab에서:
```python
!git clone https://github.com/sunnyxide/2025-inthon-baseline.git
# 또는
!cd 2025-inthon-baseline && git pull origin feature/model-evaluation-script
```

---

## ✅ 요약

1. **Clean checkpoint 생성** (Step 1 코드 실행)
2. **Model 클래스 업데이트** (Step 2 코드로 교체)
3. **평가 실행** (Step 3)

이제 RPN 키 오류 없이 정상 작동합니다! 🎉

