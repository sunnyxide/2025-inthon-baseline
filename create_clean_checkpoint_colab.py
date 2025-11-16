"""
Colab에서 바로 실행 가능한 Clean Checkpoint 생성 스크립트
- best_model.pt에서 RPN 파라미터 제거
- best_model_clean.pt 생성
"""
import torch
from pathlib import Path

def create_clean_checkpoint_colab():
    """Colab에서 clean checkpoint 생성"""
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

if __name__ == "__main__":
    create_clean_checkpoint_colab()

