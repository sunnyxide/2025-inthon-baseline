# Clean Checkpoint 가이드

> **Solution C**: RPN 파라미터를 제거한 clean checkpoint 생성 및 사용

---

## 🎯 왜 Clean Checkpoint가 필요한가?

학습 시 RPN auxiliary decoder를 사용한 모델은 체크포인트에 RPN 관련 파라미터가 포함됩니다. 하지만:

- **Inference에서는 RPN decoder가 필요 없음**
- **제출용 Model 클래스는 RPN decoder를 생성하지 않음**
- **RPN 파라미터가 있으면 `strict=True` 로딩 시 오류 발생**

**해결책**: Clean checkpoint를 생성하여 RPN 파라미터를 제거하고 inference 전용 모델로 변환

---

## 📋 사용 방법

### Step 1: Clean Checkpoint 생성

학습이 완료된 후, clean checkpoint를 생성합니다:

```bash
# 기본 사용법 (best_model_clean.pt로 저장)
python clean_checkpoint.py best_model.pt

# 출력 경로 지정
python clean_checkpoint.py best_model.pt -o inference_model.pt

# Config 없이 저장 (비추천)
python clean_checkpoint.py best_model.pt --no-config
```

**예시 출력**:
```
📂 Loading checkpoint from: best_model.pt
   Original checkpoint keys: 150
   Clean checkpoint keys: 115
   Removed RPN keys: 35
   Sample removed keys: ['rpn_decoder.layers.0.self_attn.in_proj_weight', ...]
   ✅ Kept tokenizer_config
   ✅ Removed rpn_vocab from model_config
   ✅ Kept model_config (rpn_vocab removed)

💾 Saving clean checkpoint to: best_model_clean.pt
   ✅ Clean checkpoint saved (45.23 MB)

🎉 Clean checkpoint created successfully!
   You can now use this checkpoint with Model class (strict=True)
```

### Step 2: Model 클래스 사용

`Model` 클래스는 자동으로 clean checkpoint를 우선 사용합니다:

```python
from model import Model

# best_model_clean.pt가 있으면 자동으로 사용
# 없으면 best_model.pt를 사용 (RPN 키 필터링)
model = Model()

# 예측
result = model.predict("12+34")  # "46"
```

**동작 방식**:
1. `best_model_clean.pt`가 있으면 → `strict=True`로 로드 (깔끔)
2. `best_model_clean.pt`가 없으면 → `best_model.pt`를 사용하고 RPN 키 필터링 (fallback)

---

## 🔧 코드에서 직접 사용

### Python 스크립트에서

```python
from clean_checkpoint import clean_checkpoint

# Clean checkpoint 생성
clean_checkpoint(
    input_path="best_model.pt",
    output_path="best_model_clean.pt",
    keep_config=True,
)
```

### 학습 스크립트에서 자동 저장

`train.py`에서 학습 완료 후 자동으로 clean checkpoint를 저장하려면:

```python
# train.py의 학습 완료 부분에 추가
from clean_checkpoint import clean_checkpoint

# ... 학습 완료 후 ...

# Clean checkpoint 자동 생성
clean_checkpoint(
    input_path="best_model.pt",
    output_path="best_model_clean.pt",
    keep_config=True,
)
print("✅ Clean checkpoint created: best_model_clean.pt")
```

---

## ✅ 장점

### 1. **안전성**
- `strict=True` 로딩 가능 (키 불일치 오류 없음)
- Baseline 구조와 완전히 동일
- 예상치 못한 키 문제 완전 해결

### 2. **단순성**
- 복잡한 필터링 로직 불필요
- 코드가 깔끔하고 유지보수 용이
- 평가 스크립트 변경 최소화

### 3. **효율성**
- 불필요한 RPN 파라미터 제거로 메모리 절약
- 체크포인트 파일 크기 감소
- 로딩 속도 향상

### 4. **규칙 준수**
- Inference에서는 순수 baseline 모델만 사용
- 학습용 auxiliary head 완전 분리
- 제출 규칙과 완벽히 일치

---

## 📊 비교

| 방법 | strict 로딩 | 코드 복잡도 | 메모리 | 권장도 |
|------|------------|------------|--------|--------|
| **Solution C (Clean Checkpoint)** | ✅ `strict=True` | ⭐ 단순 | ✅ 절약 | ⭐⭐⭐⭐⭐ |
| Solution A (RPN 비활성화) | ⚠️ `strict=False` | ⭐⭐ 보통 | ❌ 낭비 | ⭐⭐⭐ |
| Solution B (필터링) | ⚠️ `strict=False` | ⭐⭐⭐ 복잡 | ❌ 낭비 | ⭐⭐ |

---

## 🚨 주의사항

1. **Clean checkpoint는 inference 전용**
   - 학습 재개 시에는 원본 checkpoint 사용
   - Clean checkpoint로는 학습 불가

2. **Config 보존 권장**
   - `keep_config=True`로 설정하여 tokenizer_config, model_config 보존
   - Config가 없으면 Model 클래스 초기화 실패

3. **원본 checkpoint 백업**
   - Clean checkpoint 생성 전 원본 백업 권장
   - 필요 시 원본에서 다시 생성 가능

---

## 🔍 문제 해결

### Q: Clean checkpoint 생성 시 오류 발생

**A**: 원본 checkpoint가 유효한지 확인:
```bash
python -c "import torch; ckpt = torch.load('best_model.pt'); print('Keys:', list(ckpt.keys()))"
```

### Q: Model 클래스가 여전히 RPN 키 오류 발생

**A**: Clean checkpoint가 제대로 생성되었는지 확인:
```bash
python -c "import torch; ckpt = torch.load('best_model_clean.pt'); state = ckpt['model_state']; rpn_keys = [k for k in state.keys() if k.startswith('rpn_')]; print('RPN keys:', len(rpn_keys))"
```

### Q: Config가 없어서 Model 초기화 실패

**A**: `keep_config=True`로 clean checkpoint 재생성:
```bash
python clean_checkpoint.py best_model.pt -o best_model_clean.pt
```

---

## 📝 요약

1. **학습 완료 후**: `python clean_checkpoint.py best_model.pt`
2. **평가/제출 시**: `Model()` 클래스 자동으로 clean checkpoint 사용
3. **결과**: 깔끔하고 안전한 inference 모델 ✅

---

**이제 모든 RPN 관련 문제가 해결되었습니다!** 🎉

