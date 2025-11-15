# 로컬 개발 환경 설정 가이드

> **참고**: 이 가이드는 로컬에서 개발 환경을 설정하는 방법을 안내합니다.  
> 평가 서버와 동일한 환경을 구성하여 로컬에서 테스트할 수 있습니다.

---

## 📋 목차

1. [시작하기 전에](#시작하기-전에)
2. [자동 설정 (권장)](#자동-설정-권장)
3. [수동 설정](#수동-설정)
4. [GPU 설정](#gpu-설정)
5. [환경 확인](#환경-확인)
6. [문제 해결](#문제-해결)

---

## 시작하기 전에

### 시스템 요구사항

- **Python**: 3.10 이상
- **OS**: Linux, macOS, Windows (WSL2 권장)
- **GPU** (선택사항): NVIDIA GPU + CUDA 12.1 이상
- **메모리**: 최소 8GB RAM (GPU 사용 시 16GB 권장)

### 주의사항

⚠️ **중요**: 
- 이 가이드는 **로컬 개발 환경** 설정을 위한 것입니다.
- 평가 서버 제출 시에는 `requirements_local.txt`를 사용하지 마세요.
- 평가 서버에는 이미 필요한 라이브러리가 사전 설치되어 있습니다.

---

## 자동 설정 (권장)

가장 간단한 방법은 자동 설정 스크립트를 사용하는 것입니다.

### 1. 스크립트 실행

```bash
cd /Users/sunny/datathon/2025-inthon-baseline
bash setup_local_env.sh
```

### 2. 스크립트가 하는 일

- Python 버전 확인
- GPU/CUDA 확인
- 가상환경 생성 (선택사항)
- PyTorch 설치 (GPU/CPU 자동 감지)
- 필수 라이브러리 설치
- 설치 확인

### 3. 가상환경 활성화 (생성한 경우)

```bash
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows
```

---

## 수동 설정

자동 스크립트를 사용하지 않는 경우, 다음 단계를 따라 수동으로 설정할 수 있습니다.

### 1. 가상환경 생성 (권장)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows
```

### 2. PyTorch 설치

#### GPU 사용 가능한 경우 (NVIDIA GPU + CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

#### CPU만 사용하는 경우 (macOS 또는 GPU 없음)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

### 3. 나머지 라이브러리 설치

```bash
pip install -r requirements_local.txt
```

---

## GPU 설정

### NVIDIA GPU 확인

```bash
# GPU 정보 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version
```

### PyTorch에서 GPU 사용 확인

```bash
python3 check_gpu.py
```

또는 Python에서 직접 확인:

```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")
```

### macOS에서 GPU 사용

⚠️ **주의**: macOS는 NVIDIA GPU를 지원하지 않습니다.
- Apple Silicon (M1/M2/M3)의 경우 Metal Performance Shaders (MPS)를 사용할 수 있습니다.
- PyTorch 2.0+에서 MPS 백엔드를 지원합니다.

```python
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS (Apple Silicon GPU) 사용 가능")
```

---

## 환경 확인

설치가 완료되면 다음 명령어로 환경을 확인할 수 있습니다:

```bash
python3 check_gpu.py
```

이 스크립트는 다음을 확인합니다:
- ✅ Python 버전
- ✅ PyTorch 설치 및 버전
- ✅ CUDA 사용 가능 여부
- ✅ 필수 라이브러리 설치 여부
- ✅ 시스템 정보

### 예상 출력

```
==================================================
InThon 2025 데이터톤 환경 확인
==================================================

==================================================
Python 환경 확인
==================================================
Python 버전: 3.10.12
Python 경로: /usr/bin/python3

==================================================
PyTorch 확인
==================================================
✅ PyTorch 버전: 2.5.1+cu121
✅ CUDA 사용 가능: True
✅ CUDA 버전: 12.1
✅ GPU 개수: 1
   GPU 0: NVIDIA GeForce RTX 3090
   메모리: 24.00 GB

==================================================
필수 라이브러리 확인
==================================================
✅ NumPy: 2.2.6
✅ tqdm: 4.67.1
✅ Transformers: 4.57.1
...
```

---

## 문제 해결

### 1. PyTorch가 CUDA를 인식하지 못함

**증상**: `torch.cuda.is_available()`이 `False`를 반환

**해결 방법**:
1. NVIDIA 드라이버가 최신인지 확인:
   ```bash
   nvidia-smi
   ```
2. CUDA 버전 확인:
   ```bash
   nvcc --version
   ```
3. PyTorch 재설치 (CUDA 버전에 맞게):
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

### 2. 라이브러리 버전 충돌

**증상**: `ImportError` 또는 버전 관련 오류

**해결 방법**:
1. 가상환경 사용 (권장)
2. 모든 라이브러리 재설치:
   ```bash
   pip install --upgrade -r requirements_local.txt
   ```

### 3. 메모리 부족 오류

**증상**: `CUDA out of memory` 오류

**해결 방법**:
1. 배치 크기 줄이기 (`train.py`에서 `batch_size` 조정)
2. 모델 크기 줄이기 (`config.py`에서 `d_model` 조정)
3. CPU 모드로 실행 (GPU 없이)

### 4. macOS에서 PyTorch 설치 오류

**증상**: CUDA 버전 PyTorch 설치 실패

**해결 방법**:
- macOS는 NVIDIA GPU를 지원하지 않으므로 CPU 버전을 설치:
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  ```

### 5. 가상환경 활성화 오류

**증상**: `source venv/bin/activate` 실행 시 오류

**해결 방법**:
- Windows의 경우:
  ```bash
  venv\Scripts\activate
  ```
- PowerShell의 경우:
  ```powershell
  venv\Scripts\Activate.ps1
  ```

---

## 학습 시작하기

환경 설정이 완료되면 다음 명령어로 학습을 시작할 수 있습니다:

```bash
# 환경 확인
python3 check_gpu.py

# 학습 시작
python3 train.py

# 로컬 테스트
python3 local_test.py .
```

---

## Colab과의 차이점

### Colab 환경
- 클라우드 기반 (Google의 서버)
- GPU 자동 할당 (T4, V100 등)
- 라이브러리 사전 설치
- 세션 종료 시 데이터 삭제

### 로컬 환경
- 자신의 컴퓨터에서 실행
- GPU는 자신의 하드웨어 사용
- 라이브러리 수동 설치 필요
- 데이터 영구 보존

### 코드 호환성

Colab에서 개발한 코드는 로컬에서도 동일하게 작동해야 합니다. 다만 다음 사항을 확인하세요:

1. **경로**: Colab은 `/content/` 경로를 사용하지만, 로컬은 상대 경로 사용
2. **GPU 설정**: Colab은 자동으로 GPU를 할당하지만, 로컬은 명시적으로 설정 필요
3. **라이브러리 버전**: 동일한 버전 사용 권장

---

## 추가 리소스

- **평가 가이드**: [`EVALUATION_GUIDE.md`](./EVALUATION_GUIDE.md)
- **대회 규칙**: [`COMPETITION_RULES.md`](./COMPETITION_RULES.md)
- **모델 분석**: [`MODEL_ANALYSIS.md`](./MODEL_ANALYSIS.md)
- **PyTorch 공식 문서**: https://pytorch.org/get-started/locally/

---

**문서 작성일**: 2025년  
**최종 업데이트**: 로컬 환경 설정 가이드 작성 완료

