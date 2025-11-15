#!/bin/bash
# 로컬 개발 환경 설정 스크립트
# 평가 서버와 동일한 환경을 로컬에서 구성합니다.

set -e

echo "=========================================="
echo "InThon 2025 데이터톤 로컬 환경 설정"
echo "=========================================="
echo ""

# Python 버전 확인
echo "1. Python 버전 확인 중..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python 버전: $python_version"
echo ""

# GPU 확인 (CUDA 사용 가능 여부)
echo "2. GPU/CUDA 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "   CUDA 버전 확인 중..."
    if command -v nvcc &> /dev/null; then
        nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        echo "   CUDA 버전: $nvcc_version"
    else
        echo "   ⚠️  nvcc가 설치되어 있지 않습니다."
        echo "   PyTorch는 CUDA를 자동으로 감지할 수 있습니다."
    fi
else
    echo "   ⚠️  NVIDIA GPU가 감지되지 않았습니다."
    echo "   CPU 모드로 실행됩니다."
fi
echo ""

# 가상환경 생성 (선택사항)
echo "3. 가상환경 설정..."
read -p "   가상환경을 생성하시겠습니까? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    venv_name="venv"
    if [ ! -d "$venv_name" ]; then
        echo "   가상환경 생성 중: $venv_name"
        python3 -m venv $venv_name
        echo "   가상환경이 생성되었습니다."
    else
        echo "   가상환경이 이미 존재합니다: $venv_name"
    fi
    echo "   가상환경 활성화: source $venv_name/bin/activate"
    source $venv_name/bin/activate
else
    echo "   가상환경 생성을 건너뜁니다."
fi
echo ""

# PyTorch 설치 확인
echo "4. PyTorch 설치 확인 중..."
if python3 -c "import torch" 2>/dev/null; then
    torch_version=$(python3 -c "import torch; print(torch.__version__)")
    cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")
    echo "   PyTorch 버전: $torch_version"
    echo "   CUDA 사용 가능: $cuda_available"
    if [ "$cuda_available" = "True" ]; then
        cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
        echo "   CUDA 버전: $cuda_version"
    fi
    echo ""
    read -p "   PyTorch를 재설치하시겠습니까? (y/n): " reinstall_torch
    if [ "$reinstall_torch" != "y" ] && [ "$reinstall_torch" != "Y" ]; then
        echo "   PyTorch 설치를 건너뜁니다."
        skip_torch=true
    fi
else
    echo "   PyTorch가 설치되어 있지 않습니다."
    skip_torch=false
fi
echo ""

# PyTorch 설치 (필요한 경우)
if [ "$skip_torch" != "true" ]; then
    echo "5. PyTorch 설치 중..."
    if command -v nvidia-smi &> /dev/null; then
        echo "   GPU 버전 PyTorch 설치 중 (CUDA 12.1)..."
        pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "   CPU 버전 PyTorch 설치 중..."
        pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    fi
    echo "   PyTorch 설치 완료"
    echo ""
fi

# 나머지 라이브러리 설치
echo "6. 나머지 라이브러리 설치 중..."
pip3 install -r requirements_local.txt
echo "   라이브러리 설치 완료"
echo ""

# 최종 확인
echo "7. 설치 확인 중..."
python3 -c "
import torch
import numpy as np
import transformers
print('✅ PyTorch:', torch.__version__)
print('✅ NumPy:', np.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ CUDA 사용 가능:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA 버전:', torch.version.cuda)
    print('✅ GPU 개수:', torch.cuda.device_count())
    print('✅ GPU 이름:', torch.cuda.get_device_name(0))
"

echo ""
echo "=========================================="
echo "환경 설정이 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 명령어로 학습을 시작할 수 있습니다:"
echo "  python3 train.py"
echo ""

