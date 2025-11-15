# Colab에서 사용하기 위한 파일 목록 및 설정 가이드

## 📋 필수 파일 (복사 필요)

### 1. 핵심 데이터 로더
- **`dataloader.py`** ⭐ 필수
  - 카테고리별 데이터 생성
  - 증강 로직 포함
  - Phase별 자리수 분포 지원

### 2. 설정 파일
- **`config.py`** ⭐ 필수
  - TokenizerConfig, ModelConfig, TrainConfig 정의

### 3. 검증 및 평가 코드 (수정 금지)
- **`do_not_edit/`** 폴더 전체 ⭐ 필수
  - `dataloader_validator.py` - 데이터 검증
  - `metric.py` - 평가 지표
  - `model_template.py` - BaseModel 정의

### 4. 모델 및 학습 코드
- **`model.py`** - 모델 정의 (필요시)
- **`train.py`** - 학습 스크립트 (필요시)

---

## 🚀 Colab에서 빠른 시작

### Step 1: 필수 파일 업로드

```python
# Colab 셀에서 실행
# 1. do_not_edit 폴더 생성 및 파일 업로드
!mkdir -p do_not_edit

# 2. 필수 파일들을 Colab에 업로드하거나 직접 복사
# - dataloader.py
# - config.py
# - do_not_edit/dataloader_validator.py
# - do_not_edit/metric.py
# - do_not_edit/model_template.py
```

### Step 2: 데이터 로더 사용 예시

```python
from dataloader import ArithmeticDataset, get_dataloader

# Phase 4 (4-5자리) 데이터셋 생성
dataset = ArithmeticDataset(
    num_samples=200000,  # 샘플 수
    phase=4,             # Phase 1-4: (1-2), (2-3), (3-4), (4-5) digits
    seed=42,
    mode="train",
    enable_augmentation=True,  # expression_consistency에 증강 적용
)

# DataLoader 생성 (자동 검증 포함)
dataloader = get_dataloader(
    dataset,
    batch_size=64,
    num_workers=0,  # Colab에서는 0 권장
    pin_memory=True,
    mode="train",
)

# 사용 예시
for batch in dataloader:
    print(batch["input_text"][:5])  # 첫 5개 샘플
    print(batch["target_text"][:5])
    break
```

### Step 3: 데이터 분포 확인

```python
from collections import Counter

# 카테고리 분포 확인
categories = []
for i in range(1000):
    sample = dataset[i]
    categories.append(sample["meta"]["category"])

cat_counts = Counter(categories)
for cat, count in cat_counts.most_common():
    print(f"{cat}: {count/len(categories)*100:.1f}%")
```

---

## ⚙️ 주요 파라미터 설명

### ArithmeticDataset 파라미터

- `num_samples`: 생성할 샘플 수
- `phase`: Phase 번호 (1-4)
  - Phase 1: 1-2자리
  - Phase 2: 2-3자리
  - Phase 3: 3-4자리
  - Phase 4: 4-5자리
- `seed`: 랜덤 시드
- `mode`: "train" 또는 "val"
- `enable_augmentation`: 증강 활성화 여부 (expression_consistency에만 적용)

### 카테고리 분포 (현재 하드코딩)

```python
TRAINING_DISTRIBUTION = {
    "base_calculation": 0.40,        # 기본 사칙연산
    "precedence": 0.20,              # 괄호/우선순위
    "expression_consistency": 0.25,  # 교환/결합법칙
    "relational": 0.10,              # 관계 일관성 (A+0, A*1 등)
    "single_number": 0.05,           # 단일 숫자
}
```

### 6+ digit 출력 비율

```python
OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.05,
    "precedence": 0.10,
    "expression_consistency": 0.20,
    "relational": 0.30,
    "single_number": 0.0,
}
```

---

## 🔍 증강 로직 확인

증강은 `expression_consistency` 카테고리에만 적용되며:
- 교환법칙: `a+b` → `b+a`, `a*b` → `b*a`
- 50% 확률로 적용
- 괄호가 있는 표현식은 증강하지 않음

---

## ⚠️ 주의사항

1. **do_not_edit 폴더는 절대 수정하지 마세요**
2. **입력 숫자는 1-5자리만 허용** (규칙 준수)
3. **출력은 6자리 이상도 가능** (OOD 대비)
4. **데이터 검증은 자동으로 수행됨** (train mode에서)

---

## 📝 리뷰 포인트 (향후 개선 가능)

현재 구현은 v0 baseline이며, 다음 개선이 가능:
1. Phase별 augmentation prob 조정 (현재 50% 고정)
2. 분포를 config 파일로 분리
3. 연산자별 비율 세밀 조정
4. 길이 bin별 분포 제어

