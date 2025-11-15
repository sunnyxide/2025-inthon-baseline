# InThon 2025 데이터톤 공식 평가 가이드

> **문서 버전**: 1.0  
> **최종 업데이트**: 2025년  
> **참고**: 이 문서는 공식 평가 가이드를 기반으로 정리한 것입니다.

---

## 📋 핵심 요약

### 필수 요건

- **제출 파일**: `model.py` (또는 `main.py`)을 디렉토리 최상단에 포함해야 합니다.
- **클래스 구현**: `Model` 클래스가 `do_not_edit.model_template.BaseModel`을 상속해야 합니다.
- **메서드 구현**:
  - `__init__(self)`: 모델 로딩 등 모든 초기화
  - `predict(self, input_text: str) -> str`: 예측 수행 (반드시 문자열 반환)
- **경로**: 상대 경로만 사용 (예: `"model.pt"`, `"tokenizer.json"`)
- **제출 방법**: GitHub URL 또는 Zip 파일 (웹 앱을 통해 제출)
- **개발 환경**: 평가 서버와 동일한 Docker 이미지를 제공합니다 (`jsh0423/pytorch-cuda:12.1`)

---

## 📁 제출물 가이드

### 1. 제출 방법

#### (1) GitHub 제출

- 저장소 URL 및 선택적으로 커밋 해시를 제출
- 커밋 해시 미제출 시 최신 커밋 사용
- **Private 저장소**: Settings → Collaborators에서 `inthon` 계정 (`kucistudents`)을 Read 권한으로 추가 (출제 팀장에게 문의)

#### (2) Zip 파일 제출

- 전체 디렉토리를 압축하여 웹 앱에 업로드

### 2. 필수 제출 내용

⚠️ **중요**: 모든 제출물에는 다음이 반드시 포함되어야 합니다:

#### 모델 실행 파일

- `model.py` (또는 `main.py`)에 `BaseModel`을 상속한 `Model` 클래스 구현
- 체크포인트는 상대 경로로 로드 (예: `"best_model.pt"`)

#### 학습 코드 (Training Code)

- 실험이 완전히 재현 가능해야 함
- `.sh` 파일 또는 상세한 `README.md`로 실행 방법 명시
- 사용한 하이퍼파라미터, 학습 환경 등 기록

#### 모델 가중치

- 100MB 초과 시 Git LFS 사용 권장
- Zip 제출 시 직접 포함 가능

#### `do_not_edit/` 폴더 (Baseline 코드에서 제공)

- `model_template.py`, `dataloader_validator.py`, `metric.py` 포함
- **절대 수정 금지**

### 3. 디렉토리 구조 예시

```
my_team/
├── model.py              # 필수: Model 클래스 구현
├── best_model.pt         # 학습된 가중치
├── train.py              # 필수: 학습 코드
├── run_train.sh          # 권장: 재현 스크립트
├── README.md             # 권장: 실행 방법 및 환경 설명
├── tokenizer.json        # 필요 시
├── config.py             # 필요 시
├── requirements.txt      # 추가 라이브러리 (선택)
├── do_not_edit/          # 필수: Baseline에서 제공
│   ├── model_template.py
│   ├── dataloader_validator.py
│   └── metric.py
└── .gitattributes        # Git LFS 사용 시
```

---

## 📚 Baseline 코드 참고

여러분들의 원활한 시작을 돕기 위해 공식 Baseline 코드를 GitHub에서 제공합니다:

1. **로컬/서버 개발 환경 기반**: [InThon Datathon Baseline](https://github.com/kucistudents/2025-inthon-baseline)
2. **Colab / Kaggle 등 Notebook 기반**: [InThon Datathon Baseline](https://github.com/kucistudents/2025-inthon-baseline)

**포함 내용**:
- 간단한 GRU 기반 Seq2Seq 모델
- 토크나이저
- 데이터 로더
- 학습 스크립트
- `do_not_edit/` 폴더: 수정하지 않는 것을 권장 (BaseModel 정의, 검증기, 평가 지표 포함)

---

## 🚀 모델 구현 (필수 사항)

`model.py`에 `BaseModel`을 상속하여 `Model` 클래스를 구현해야 합니다.

모델의 세부 구조는 외부 파일로 자유롭게 분리하여 import 가능합니다.

평가 시스템은 `Model()` 인스턴스를 생성한 후 `predict()` 메서드를 호출합니다.

### 구현 예시

```python
from do_not_edit.model_template import BaseModel

class Model(BaseModel):
    def __init__(self):
        """모든 초기화를 여기서 완료 (모델 로딩, 토크나이저 등)"""
        super().__init__()
        # 상대 경로로 모델 로드
        # self.model = torch.load("best_model.pt")

    def predict(self, input_text: str) -> str:
        """입력 문자열 → 예측 결과 문자열 반환 (필수)"""
        # 실제 예측 로직
        return "42"
```

### ⚠️ 핵심 규칙

- `BaseModel`을 반드시 상속
- `__init__()`은 인자 없이 호출 가능해야 함
- `predict(input_text: str) -> str` 구현 필수
- 상대 경로만 사용 (예: `"model.pt"`, `"tokenizer.json"`)
- 항상 문자열을 반환

---

## 🐳 로컬 개발 환경 (Docker 이미지 제공)

평가 서버와 동일한 환경에서 개발할 수 있도록 Docker 이미지를 제공합니다!

### Docker 이미지 다운로드 및 실행

```bash
# 이미지 다운로드 (최초 1회)
docker pull jsh0423/pytorch-cuda:12.1

# 기본 사용 (Linux/WSL2 - GPU 지원)
docker run --gpus all -it --rm -v $(pwd):/workspace jsh0423/pytorch-cuda:12.1
```

### 포함된 라이브러리

- **PyTorch**: 2.5.1+cu121 (CUDA 12.1)
- **torchvision**: 0.20.1+cu121
- **torchaudio**: 2.5.1+cu121
- **NumPy**: 2.2.6
- **tqdm**: 4.67.1
- **Transformers**: 4.57.1
- **Tokenizers**: 0.22.1
- **SentencePiece**: 0.2.1
- **Accelerate**: 1.11.0
- **einops**: 0.8.1
- **safetensors**: 0.6.2
- **regex**: 2025.11.3
- **pydantic**: 2.12.4
- **Jupyter, Matplotlib, Pandas** (개발 편의용)

**참고**: 이 이미지는 평가 환경과 동일한 라이브러리를 포함하므로, 로컬에서 정상 동작하면 서버에서도 동일하게 작동합니다.

---

## 📦 대용량 파일 (Git LFS)

모델 가중치 파일이 50MB 이상이거나 GitHub의 100MB 제한을 초과하는 경우 Git LFS를 권장합니다.

### Git LFS 설정 방법

```bash
# 1. Git LFS 설치 및 초기화
git lfs install

# 2. LFS로 추적할 파일 확장자 지정
git lfs track "*.pt"
git lfs track "*.bin"
git lfs track "*.safetensors"

# 3. .gitattributes 파일 커밋 (필수)
git add .gitattributes

# 4. 대용량 파일 추가, 커밋, 푸시
git add model.pt
git commit -m "Add model weights"
git push
```

---

## 🖥️ 평가 환경 상세

제출된 모델은 격리된 Docker 컨테이너에서 실행됩니다.

### 1. 실행 환경

- **Base Image**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10+
- **작업 디렉토리**: 제출된 디렉토리 (예: `/workspace/my_team`)

### 2. 사전 설치된 라이브러리

다음 라이브러리는 평가 서버에 기본으로 설치되어 있습니다. **`requirements.txt`에 포함하지 마세요.**

#### 핵심 라이브러리

```
torch==2.5.1+cu121              # PyTorch (CUDA 12.1)
torchvision==0.20.1+cu121       # 이미지 처리 (torch 호환 버전)
torchaudio==2.5.1+cu121         # 오디오 처리 (torch 호환 버전)
numpy==2.2.6                    # 수치 계산
tqdm==4.67.1                    # Progress bar
```

#### NLP 및 모델 라이브러리

```
transformers==4.57.1            # Hugging Face Transformers
tokenizers==0.22.1              # 빠른 커스텀 토크나이저
sentencepiece==0.2.1            # 서브워드 토크나이저
```

#### 최적화 및 유틸리티

```
accelerate==1.11.0              # 추론 가속 및 메모리 최적화
einops==0.8.1                   # 텐서 조작 편의성
safetensors==0.6.2              # 빠르고 안전한 모델 저장/로딩
regex==2025.11.3                # 정규표현식
pydantic==2.12.4                # 데이터 검증
```

### 3. requirements.txt (선택 사항)

추가 라이브러리가 필요한 경우 `requirements.txt`를 제출하면 자동 설치됩니다.

⚠️ **주의**:
- 사전 설치된 라이브러리는 포함하지 마세요 (충돌 방지)
- 설치 시간도 30분 제한에 포함됩니다

### 4. 리소스 제약

| 항목 | 제약사항 |
|------|----------|
| **인터넷** | 차단 (모든 파일은 제출물에 포함) |
| **CPU** | 2 코어 (워크로드 프로필: Consumption-GPU-NC8as-T4) |
| **메모리** | 4Gi |
| **GPU** | NVIDIA T4 1개, VRAM 16GB, 단일 GPU (`CUDA_VISIBLE_DEVICES=0`) |
| **타임아웃** | 3600초 (1시간) |

---

## ✅ 제출 전 확인사항

제출하기 전에 다음 항목을 반드시 확인하세요:

### 필수 확인사항

- [ ] `model.py`에 `Model` 클래스가 `BaseModel`을 상속하고 있는가?
- [ ] `__init__()`에서 체크포인트를 상대 경로로 로드하는가?
- [ ] `predict()` 메서드가 문자열을 반환하는가?
- [ ] 학습 코드(`train.py`)가 포함되어 있는가?
- [ ] 실험 재현이 가능한가? (`.sh` 파일 또는 `README.md`, 내지는 재현 가능한 기본 값 등)
- [ ] `do_not_edit/` 폴더가 수정되지 않은 채로 포함되어 있는가?
- [ ] 100MB 이상 파일은 Git LFS로 관리되고 있는가?
- [ ] Private 저장소의 경우 `inthon-ku` 계정이 Collaborator로 추가되었는가?

### 추가 확인사항

- [ ] 모든 경로가 상대 경로로 설정되어 있는가?
- [ ] `requirements.txt`에 사전 설치된 라이브러리가 포함되지 않았는가?
- [ ] 모델이 1시간 내에 초기화 및 예측을 완료할 수 있는가?
- [ ] GPU 메모리 사용량이 16GB를 초과하지 않는가?

---

## 📝 제출 체크리스트

### 파일 구조

```
[ ] model.py (또는 main.py) - 최상단에 위치
[ ] train.py - 학습 코드
[ ] best_model.pt (또는 다른 체크포인트 파일)
[ ] do_not_edit/ - 수정되지 않은 원본
    [ ] model_template.py
    [ ] dataloader_validator.py
    [ ] metric.py
[ ] README.md - 실행 방법 설명 (권장)
[ ] run_train.sh - 재현 스크립트 (권장)
[ ] requirements.txt - 추가 라이브러리 (필요 시)
[ ] .gitattributes - Git LFS 사용 시
```

### 코드 검증

```
[ ] Model 클래스가 BaseModel 상속
[ ] __init__() 메서드가 인자 없이 호출 가능
[ ] predict(input_text: str) -> str 메서드 구현
[ ] 모든 파일 경로가 상대 경로
[ ] predict()가 항상 문자열 반환
```

### 재현성

```
[ ] 학습 스크립트 실행 가능
[ ] 하이퍼파라미터 문서화
[ ] 환경 설정 방법 문서화
[ ] 시드 값 고정 (선택)
```

---

## 🔗 참고 자료

- **📋 대회 규칙 문서**: [`COMPETITION_RULES.md`](./COMPETITION_RULES.md) - 규칙, 제약사항, 금지 행위 등
- **대회 공식 플랫폼**: https://jolly-bush-05d5dc000.3.azurestaticapps.net/index.html
- **대회 규칙 및 평가 가이드**: https://jolly-bush-05d5dc000.3.azurestaticapps.net/rule.html
- **Baseline 코드**: https://github.com/kucistudents/2025-inthon-baseline
- **문의**: 출제팀장(전성후)

---

**문서 작성일**: 2025년  
**최종 확인**: 공식 평가 가이드 반영 완료

