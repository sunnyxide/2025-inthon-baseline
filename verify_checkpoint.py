#!/usr/bin/env python3
"""
Complete Checkpoint Verification Script
체크포인트를 로드하기 전에 이 스크립트로 검증하세요.

Usage:
    python verify_checkpoint.py                    # best_model.pt 검증
    python verify_checkpoint.py checkpoint.pt      # 특정 파일 검증
"""

import sys
import torch
import os
from config import ModelConfig, TrainConfig, TokenizerConfig
from model import TransformerSeq2Seq, CharTokenizer, INPUT_CHARS, OUTPUT_CHARS


def verify_checkpoint(checkpoint_path="best_model.pt", run_inference_test=True):
    """
    체크포인트 완전 검증
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        run_inference_test: 추론 테스트 실행 여부
    
    Returns:
        bool: 검증 성공 여부
    """
    
    print("=" * 70)
    print("🔍 CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    # ========================================================================
    # 1. 파일 존재 확인
    # ========================================================================
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✅ Checkpoint found: {checkpoint_path}")
    file_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"📦 Size: {file_size_mb:.2f} MB\n")
    
    # ========================================================================
    # 2. 체크포인트 로드
    # ========================================================================
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    # ========================================================================
    # 3. 내용 확인
    # ========================================================================
    print("📝 Checkpoint Contents:")
    required_keys = ["model_state", "model_config", "train_config"]
    optional_keys = ["optim_state", "step", "tokenizer_config"]
    
    for key in ckpt.keys():
        if key == "model_state":
            num_params = len(ckpt[key])
            print(f"  ✅ {key:20s}: {num_params} parameters")
        elif key == "optim_state":
            print(f"  ✅ {key:20s}: optimizer state saved")
        elif key == "step":
            print(f"  ✅ {key:20s}: {ckpt[key]}")
        elif key in ["model_config", "train_config", "tokenizer_config"]:
            print(f"  ✅ {key:20s}: configuration saved")
        else:
            print(f"  ℹ️  {key:20s}: {type(ckpt[key])}")
    
    # 필수 키 확인
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        print(f"\n⚠️  WARNING: Missing required keys: {missing_keys}")
    print()
    
    # ========================================================================
    # 4. 설정 비교 (train.py의 current config와 비교)
    # ========================================================================
    if "model_config" not in ckpt:
        print("⚠️  WARNING: model_config not found in checkpoint")
        print("Cannot verify configuration match\n")
    else:
        saved_model_cfg = ckpt["model_config"]
        
        # train.py의 W&B sweep 최적값 (현재 기본 설정)
        current_model_cfg = {
            "d_model": 256,
            "nhead": 2,
            "num_encoder_layers": 6,
            "num_decoder_layers": 2,
            "dim_feedforward": 1024,
            "dropout": 0.0,
        }
        
        print("🔧 Model Configuration Comparison:")
        print(f"{'Parameter':25s} {'Current':>10s} {'Saved':>10s} {'Status':>8s}")
        print("-" * 60)
        
        all_match = True
        for key in current_model_cfg.keys():
            current = current_model_cfg[key]
            saved = saved_model_cfg.get(key, "N/A")
            
            if saved == "N/A":
                status = "⚠️ MISSING"
                all_match = False
            elif current == saved:
                status = "✅ MATCH"
            else:
                status = "❌ DIFF"
                all_match = False
            
            print(f"{key:25s} {str(current):>10s} {str(saved):>10s} {status:>8s}")
        
        print()
        if all_match:
            print("✅ ✅ ✅ ALL CONFIGURATIONS MATCH! ✅ ✅ ✅\n")
        else:
            print("⚠️  WARNING: CONFIGURATION MISMATCH DETECTED!")
            print("   The checkpoint may not load correctly with current settings.")
            print("   Consider updating train.py or using a different checkpoint.\n")
    
    # ========================================================================
    # 5. Weight 로드 테스트
    # ========================================================================
    print("🔄 Testing Weight Loading...")
    
    try:
        # Tokenizers 준비
        input_tokenizer = CharTokenizer(INPUT_CHARS, add_special=True)
        output_tokenizer = CharTokenizer(OUTPUT_CHARS, add_special=True)
        
        # 저장된 설정 또는 기본 설정 사용
        if "model_config" in ckpt:
            model_cfg = ckpt["model_config"]
        else:
            print("⚠️  Using default configuration (checkpoint config not found)")
            model_cfg = {
                "d_model": 256,
                "nhead": 2,
                "num_encoder_layers": 6,
                "num_decoder_layers": 2,
                "dim_feedforward": 1024,
                "dropout": 0.0,
            }
        
        # 모델 생성
        model = TransformerSeq2Seq(
            in_vocab=input_tokenizer.vocab_size,
            out_vocab=output_tokenizer.vocab_size,
            **model_cfg
        )
        
        # Weights 로드 전 샘플 저장
        random_weight_sample = model.encoder.layers[0].linear1.weight[0, :3].clone()
        
        # Weights 로드
        model.load_state_dict(ckpt["model_state"])
        print("✅ Weights loaded successfully")
        
        # Weights 로드 후 샘플 확인
        loaded_weight_sample = model.encoder.layers[0].linear1.weight[0, :3]
        
        if torch.allclose(random_weight_sample, loaded_weight_sample):
            print("❌ ERROR: Weights unchanged! Load may have failed.")
            return False
        else:
            print("✅ Weights successfully updated from checkpoint")
        
        # Weight 샘플 표시
        print(f"\n📊 Weight Sample (encoder.layers[0].linear1.weight[0, :3]):")
        print(f"   {loaded_weight_sample.tolist()}")
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # 6. 추론 테스트 (선택적)
    # ========================================================================
    if run_inference_test:
        print("\n🧪 Running Inference Test...")
        
        try:
            model.eval()
            device = torch.device("cpu")
            
            test_cases = [
                ("2+3", "5"),
                ("10-5", "5"),
                ("3*4", "12"),
                ("(2+3)*4", "20"),
            ]
            
            print(f"\n{'Expression':15s} {'Predicted':>12s} {'Expected':>12s} {'Status':>8s}")
            print("-" * 55)
            
            passed = 0
            with torch.no_grad():
                for expr, expected in test_cases:
                    # Tokenize
                    src_tokens = [input_tokenizer.stoi.get(c, input_tokenizer.pad_id) 
                                  for c in expr]
                    src = torch.tensor([src_tokens], dtype=torch.long).to(device)
                    
                    # Generate
                    gen_ids = model.generate(
                        src=src,
                        max_len=24,
                        bos_id=output_tokenizer.bos_id,
                        eos_id=output_tokenizer.eos_id,
                        src_pad_id=input_tokenizer.pad_id,
                    )
                    
                    # Decode
                    result = []
                    for t in gen_ids[0].tolist():
                        if t == output_tokenizer.eos_id:
                            break
                        if t in output_tokenizer.itos:
                            ch = output_tokenizer.itos[t]
                            if ch.isdigit() or ch == '-':
                                result.append(ch)
                    
                    pred = "".join(result) if result else "ERROR"
                    status = "✅ PASS" if pred == expected else "❌ FAIL"
                    
                    if pred == expected:
                        passed += 1
                    
                    print(f"{expr:15s} {pred:>12s} {expected:>12s} {status:>8s}")
            
            print()
            print(f"📊 Inference Test Results: {passed}/{len(test_cases)} passed")
            
            if passed == len(test_cases):
                print("✅ All inference tests passed!")
            elif passed > 0:
                print("⚠️  Some tests failed - model may need more training")
            else:
                print("❌ All tests failed - model may not be trained properly")
            print()
            
        except Exception as e:
            print(f"❌ Inference test failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 7. 최종 요약
    # ========================================================================
    print("=" * 70)
    print("✅ ✅ ✅ VERIFICATION COMPLETE ✅ ✅ ✅")
    print("=" * 70)
    print()
    print("📋 Summary:")
    print(f"  • Checkpoint file:   {checkpoint_path}")
    print(f"  • File size:         {file_size_mb:.2f} MB")
    print(f"  • Configuration:     {'✅ Match' if all_match else '⚠️  Mismatch'}")
    print(f"  • Weights loaded:    ✅ Success")
    if run_inference_test:
        print(f"  • Inference test:    {passed}/{len(test_cases)} passed")
    print()
    
    if all_match:
        print("🚀 Ready to resume training! Set resume_from_checkpoint=True in train.py")
    else:
        print("⚠️  Configuration mismatch detected.")
        print("   Update train.py to match checkpoint config, or train from scratch.")
    
    print("=" * 70)
    
    return True


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify checkpoint file")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="best_model.pt",
        help="Path to checkpoint file (default: best_model.pt)"
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip inference test"
    )
    
    args = parser.parse_args()
    
    success = verify_checkpoint(
        checkpoint_path=args.checkpoint,
        run_inference_test=not args.no_inference
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

