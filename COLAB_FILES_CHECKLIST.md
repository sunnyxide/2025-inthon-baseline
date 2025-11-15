# 🚀 Colab 복사-붙여넣기 체크리스트

## ✅ 증강 구현 확인

**✓ 증강이 제대로 구현되어 있습니다:**
- `_safe_augment_expression()` 함수 구현됨 (line 229-264)
- `expression_consistency` 카테고리에만 적용
- 교환법칙만 구현 (a+b → b+a, a*b → b*a)
- Phase별 증강 확률 조정 가능 (리뷰 반영)

---

## 📁 Colab에 복사해야 할 파일 목록

### ⭐ 필수 파일 (반드시 복사)

1. **`dataloader.py`** 
   - 카테고리별 데이터 생성
   - 증강 로직 포함
   - Phase별 자리수 분포 및 증강 확률 지원

2. **`config.py`**
   - TokenizerConfig, ModelConfig, TrainConfig 정의

3. **`do_not_edit/` 폴더 전체**
   ```
   do_not_edit/
   ├── dataloader_validator.py  # 데이터 검증 (필수)
   ├── metric.py                # 평가 지표 (필수)
   └── model_template.py        # BaseModel 정의 (필수)
   ```

### 📝 선택 파일 (필요시)

4. **`model.py`** - 모델 정의 (TinySeq2Seq, Transformer 등)
5. **`train.py`** - 학습 스크립트

---

## 🔧 리뷰 반영 사항

### ✅ 적용 완료

1. **Phase별 증강 확률 조정**
   - Phase 1: 5% (기본기 안정화)
   - Phase 2: 10%
   - Phase 3: 25% (증강 강화)
   - Phase 4: 15% (상기 정도)
   - 기존: 50% 고정 → 개선됨

### 📋 향후 개선 가능 (현재 구조 유지하며 적용 가능)

1. **분포를 config로 분리** (현재는 하드코딩)
   - `TRAINING_DISTRIBUTION`, `OUTPUT_6DIGIT_RATIO` 등을 config 파일로 이동 가능
   - 하지만 현재 구조도 충분히 사용 가능

2. **연산자별 비율 세밀 조정**
   - `_gen_base_calculation()` 내부의 `op_weights`를 파라미터로 받도록 확장 가능
   - 현재는 `{"+": 0.40, "-": 0.25, "*": 0.25, "//": 0.10}` 고정

3. **길이 bin별 분포 제어**
   - Phase 내에서도 식 길이별 분포를 더 세밀하게 제어 가능
   - 현재는 자리수만 제어

---

## 📋 Colab 사용 예시

```python
# 1. 필수 파일 업로드 후
from dataloader import ArithmeticDataset, get_dataloader

# 2. 데이터셋 생성
dataset = ArithmeticDataset(
    num_samples=200000,
    phase=4,  # 4-5자리
    seed=42,
    mode="train",
    enable_augmentation=True,
)

# 3. DataLoader 생성
dataloader = get_dataloader(
    dataset,
    batch_size=64,
    num_workers=0,  # Colab에서는 0 권장
    pin_memory=True,
    mode="train",
)

# 4. 사용
for batch in dataloader:
    print(batch["input_text"][:5])
    print(batch["target_text"][:5])
    break
```

---

## ⚠️ 주의사항

1. **do_not_edit 폴더는 절대 수정 금지**
2. **입력 숫자는 1-5자리만 허용** (규칙 준수)
3. **출력은 6자리 이상도 가능** (OOD 대비)
4. **데이터 검증은 자동 수행** (train mode에서)

---

## 🎯 현재 구현 상태

- ✅ 증강 구현 완료
- ✅ Phase별 증강 확률 조정 (리뷰 반영)
- ✅ 카테고리별 분포 구현
- ✅ 6+ digit 출력 비율 제어
- ✅ 데이터 검증 통합
- 📋 분포 config화 (향후 개선 가능)
- 📋 연산자별 세밀 조정 (향후 개선 가능)

**결론: 현재 상태로도 Colab에서 바로 사용 가능하며, 리뷰의 핵심 포인트(Phase별 증강 확률)는 반영 완료**

