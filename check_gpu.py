#!/usr/bin/env python3
"""
GPU 및 환경 확인 스크립트
로컬 환경에서 GPU 사용 가능 여부와 라이브러리 버전을 확인합니다.
"""

import sys

def check_python_version():
    """Python 버전 확인"""
    print("=" * 50)
    print("Python 환경 확인")
    print("=" * 50)
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}")
    print()

def check_pytorch():
    """PyTorch 설치 및 GPU 확인"""
    print("=" * 50)
    print("PyTorch 확인")
    print("=" * 50)
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        print(f"✅ CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA 버전: {torch.version.cuda}")
            print(f"✅ cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"✅ GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            print("   GPU를 사용하려면 NVIDIA GPU와 CUDA 드라이버가 필요합니다.")
        print()
        return True
    except ImportError:
        print("❌ PyTorch가 설치되어 있지 않습니다.")
        print("   다음 명령어로 설치하세요:")
        print("   pip install torch torchvision torchaudio")
        print()
        return False

def check_libraries():
    """필수 라이브러리 확인"""
    print("=" * 50)
    print("필수 라이브러리 확인")
    print("=" * 50)
    
    libraries = {
        "numpy": "NumPy",
        "tqdm": "tqdm",
        "transformers": "Transformers",
        "tokenizers": "Tokenizers",
        "sentencepiece": "SentencePiece",
        "accelerate": "Accelerate",
        "einops": "einops",
        "safetensors": "safetensors",
    }
    
    missing = []
    for module, name in libraries.items():
        try:
            lib = __import__(module)
            version = getattr(lib, "__version__", "unknown")
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 설치되지 않음")
            missing.append(name)
    
    if missing:
        print()
        print("⚠️  다음 라이브러리가 설치되지 않았습니다:")
        for lib in missing:
            print(f"   - {lib}")
        print()
        print("다음 명령어로 설치하세요:")
        print("   pip install -r requirements_local.txt")
    print()

def check_system():
    """시스템 정보 확인"""
    print("=" * 50)
    print("시스템 정보")
    print("=" * 50)
    
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"아키텍처: {platform.machine()}")
    print()
    
    # NVIDIA GPU 확인
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("NVIDIA GPU 정보:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        print(f"   GPU: {parts[0]}")
                        print(f"   드라이버: {parts[1]}")
                        print(f"   메모리: {parts[2]}")
        else:
            print("⚠️  nvidia-smi를 실행할 수 없습니다.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  NVIDIA GPU가 감지되지 않았거나 nvidia-smi가 설치되지 않았습니다.")
    print()

def main():
    """메인 함수"""
    print("\n" + "=" * 50)
    print("InThon 2025 데이터톤 환경 확인")
    print("=" * 50 + "\n")
    
    check_python_version()
    pytorch_ok = check_pytorch()
    check_libraries()
    check_system()
    
    print("=" * 50)
    if pytorch_ok:
        print("✅ 환경 확인 완료!")
        print("=" * 50)
        print("\n다음 명령어로 학습을 시작할 수 있습니다:")
        print("  python3 train.py")
    else:
        print("⚠️  PyTorch를 먼저 설치해주세요.")
        print("=" * 50)
        print("\n환경 설정 스크립트를 실행하세요:")
        print("  bash setup_local_env.sh")
    print()

if __name__ == "__main__":
    main()

