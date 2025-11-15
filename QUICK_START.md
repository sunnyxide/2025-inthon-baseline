# 빠른 시작 가이드

> **목표**: 최소한의 변경으로 최대 성능 향상

---

## 🚀 3단계로 시작하기

### Step 1: 모델 변경 (model.py)

1. `model.py` 파일을 열기
2. `BahdanauAttention` 클래스 추가 (CODE_CHANGES.md 참고)
3. `TinySeq2Seq` 클래스를 `AttentionSeq2Seq`로 교체 (CODE_CHANGES.md 참고)
4. `Model` 클래스에서 `TinySeq2Seq` → `AttentionSeq2Seq` 변경

### Step 2: 설정 변경 (config.py)

1. `config.py` 파일을 열기
2. `ModelConfig`에 새로운 하이퍼파라미터 추가:
   ```python
   d_model: int = 512
   num_encoder_layers: int = 2
   num_decoder_layers: int = 2
   dropout: float = 0.1
   ```
3. `TrainConfig`에 추가:
   ```python
   beam_size: int = 5
   teacher_forcing_ratio: float = 0.9
   use_scheduled_sampling: bool = True
   ```

### Step 3: 학습 코드 변경 (train.py)

1. `train.py` 파일을 열기
2. import 문 변경: `TinySeq2Seq` → `AttentionSeq2Seq`
3. 모델 생성 부분 변경: `TinySeq2Seq` → `AttentionSeq2Seq`
4. Scheduled Sampling 로직 추가 (CODE_CHANGES.md 참고)
5. `generate()` 호출에 `beam_size` 추가

---

## 📝 체크리스트

변경 전에 확인:

- [ ] `model.py`에 `BahdanauAttention` 클래스 추가됨
- [ ] `model.py`에 `AttentionSeq2Seq` 클래스 추가됨
- [ ] `model.py`의 `Model` 클래스에서 `AttentionSeq2Seq` 사용
- [ ] `config.py`의 `ModelConfig` 업데이트됨
- [ ] `config.py`의 `TrainConfig` 업데이트됨
- [ ] `train.py`의 import 문 변경됨
- [ ] `train.py`의 모델 생성 부분 변경됨
- [ ] `train.py`에 Scheduled Sampling 추가됨
- [ ] `train.py`에 Beam Search 추가됨

---

## 🎯 예상 결과

- **현재 EM**: 0.40
- **예상 EM**: 0.65~0.75 (Phase 1 완료 시)
- **최종 목표 EM**: 0.85+ (모든 Phase 완료 시)

---

## 📚 상세 가이드

- **전략 가이드**: [`OPTIMAL_MODEL_GUIDE.md`](./OPTIMAL_MODEL_GUIDE.md)
- **코드 변경사항**: [`CODE_CHANGES.md`](./CODE_CHANGES.md)
- **모델 분석**: [`MODEL_ANALYSIS.md`](./MODEL_ANALYSIS.md)

---

**Good luck! 🚀**

