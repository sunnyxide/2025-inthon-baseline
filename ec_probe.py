#!/usr/bin/env python3
"""
EC (Expression Consistency) Probe Test
교환법칙/결합법칙 일관성을 직접 측정하는 테스트

Usage:
    python ec_probe.py best_model.pt
    python ec_probe.py best_model_ec.pt
"""

import sys
import torch
from model import TransformerSeq2Seq, CharTokenizer, INPUT_CHARS, OUTPUT_CHARS, tokenize_batch
from dataloader import ArithmeticDataset, _safe_augment_expression
import random


def generate_ec_probe_dataset(num_samples=500, seed=9999):
    """
    EC 검증용 probe dataset 생성.
    각 샘플은 (원본, 변형) 쌍으로 구성되며, 둘다 같은 정답을 가져야 함.
    """
    probe_pairs = []
    rng = random.Random(seed)
    
    # ArithmeticDataset에서 expression_consistency 카테고리만 생성
    dataset = ArithmeticDataset(
        num_samples=num_samples * 3,  # 충분히 생성해서 필터링
        phase=2,
        seed=seed,
        mode="train",
        enable_augmentation=False,  # augmentation 없이 원본만
    )
    
    collected = 0
    for i in range(len(dataset)):
        if collected >= num_samples:
            break
            
        sample = dataset[i]
        
        # expression_consistency 카테고리만 선택
        if sample["meta"]["category"] != "expression_consistency":
            continue
        
        expr_orig = sample["input_text"]
        val = int(sample["target_text"])
        
        # Augmentation 적용
        augmented = _safe_augment_expression(expr_orig, val, rng)
        if augmented is None:
            continue
        
        expr_aug, val_aug = augmented
        if val != val_aug:
            continue  # 안전성 체크
        
        probe_pairs.append({
            "expr_orig": expr_orig,
            "expr_aug": expr_aug,
            "value": val,
        })
        collected += 1
    
    return probe_pairs


def evaluate_ec_consistency(model, probe_pairs, tokenizers, device):
    """
    EC 일관성 평가.
    원본과 augmented 표현 모두 같은 답을 내는지 확인.
    """
    input_tokenizer, output_tokenizer = tokenizers
    model.eval()
    
    results = {
        "both_correct": 0,
        "orig_only": 0,
        "aug_only": 0,
        "both_wrong": 0,
        "total": len(probe_pairs),
    }
    
    samples_to_show = []
    
    with torch.no_grad():
        for pair in probe_pairs:
            expr_orig = pair["expr_orig"]
            expr_aug = pair["expr_aug"]
            expected = str(pair["value"])
            
            # 원본 예측
            batch_orig = {"input_text": [expr_orig], "target_text": ["0"]}
            bt_orig = tokenize_batch(batch_orig, input_tokenizer, output_tokenizer)
            src_orig = bt_orig.src.to(device)
            
            gen_orig = model.generate(
                src=src_orig,
                max_len=50,
                bos_id=output_tokenizer.bos_id,
                eos_id=output_tokenizer.eos_id,
                src_pad_id=input_tokenizer.pad_id,
            )
            
            pred_orig = ""
            for t in gen_orig[0].tolist():
                if t == output_tokenizer.eos_id:
                    break
                if t in output_tokenizer.itos:
                    ch = output_tokenizer.itos[t]
                    if ch.isdigit():
                        pred_orig += ch
            
            # Augmented 예측
            batch_aug = {"input_text": [expr_aug], "target_text": ["0"]}
            bt_aug = tokenize_batch(batch_aug, input_tokenizer, output_tokenizer)
            src_aug = bt_aug.src.to(device)
            
            gen_aug = model.generate(
                src=src_aug,
                max_len=50,
                bos_id=output_tokenizer.bos_id,
                eos_id=output_tokenizer.eos_id,
                src_pad_id=input_tokenizer.pad_id,
            )
            
            pred_aug = ""
            for t in gen_aug[0].tolist():
                if t == output_tokenizer.eos_id:
                    break
                if t in output_tokenizer.itos:
                    ch = output_tokenizer.itos[t]
                    if ch.isdigit():
                        pred_aug += ch
            
            # 결과 분류
            orig_correct = (pred_orig == expected)
            aug_correct = (pred_aug == expected)
            
            if orig_correct and aug_correct:
                results["both_correct"] += 1
            elif orig_correct and not aug_correct:
                results["orig_only"] += 1
                if len(samples_to_show) < 5:
                    samples_to_show.append({
                        "type": "orig_only",
                        "orig": expr_orig,
                        "aug": expr_aug,
                        "expected": expected,
                        "pred_orig": pred_orig,
                        "pred_aug": pred_aug,
                    })
            elif not orig_correct and aug_correct:
                results["aug_only"] += 1
                if len(samples_to_show) < 5:
                    samples_to_show.append({
                        "type": "aug_only",
                        "orig": expr_orig,
                        "aug": expr_aug,
                        "expected": expected,
                        "pred_orig": pred_orig,
                        "pred_aug": pred_aug,
                    })
            else:
                results["both_wrong"] += 1
    
    # 지표 계산
    total = results["total"]
    ec_consistency = results["both_correct"] / total if total > 0 else 0
    orig_accuracy = (results["both_correct"] + results["orig_only"]) / total if total > 0 else 0
    aug_accuracy = (results["both_correct"] + results["aug_only"]) / total if total > 0 else 0
    
    return {
        "results": results,
        "ec_consistency": ec_consistency,
        "orig_accuracy": orig_accuracy,
        "aug_accuracy": aug_accuracy,
        "samples": samples_to_show,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ec_probe.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    print("=" * 70)
    print("🔬 EC (Expression Consistency) Probe Test")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Setup tokenizers
    input_tokenizer = CharTokenizer(INPUT_CHARS, add_special=True)
    output_tokenizer = CharTokenizer(OUTPUT_CHARS, add_special=True)
    
    # Load model
    model_cfg = checkpoint.get("model_config", {
        "d_model": 256,
        "nhead": 2,
        "num_encoder_layers": 6,
        "num_decoder_layers": 2,
        "dim_feedforward": 1024,
        "dropout": 0.0,
    })
    
    model = TransformerSeq2Seq(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        **model_cfg
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Model loaded successfully\n")
    
    # Generate probe dataset
    print("📋 Generating EC probe dataset...")
    probe_pairs = generate_ec_probe_dataset(num_samples=500)
    print(f"✅ Generated {len(probe_pairs)} probe pairs\n")
    
    # Evaluate
    print("🧪 Running EC consistency test...")
    eval_results = evaluate_ec_consistency(
        model, probe_pairs, (input_tokenizer, output_tokenizer), device
    )
    
    # Print results
    print("=" * 70)
    print("📊 EC Probe Results")
    print("=" * 70)
    
    res = eval_results["results"]
    print(f"Total pairs:        {res['total']}")
    print(f"Both correct:       {res['both_correct']:4d} ({res['both_correct']/res['total']*100:.1f}%)")
    print(f"Orig only:          {res['orig_only']:4d} ({res['orig_only']/res['total']*100:.1f}%)")
    print(f"Aug only:           {res['aug_only']:4d} ({res['aug_only']/res['total']*100:.1f}%)")
    print(f"Both wrong:         {res['both_wrong']:4d} ({res['both_wrong']/res['total']*100:.1f}%)")
    print()
    print(f"🎯 EC Consistency:  {eval_results['ec_consistency']*100:.2f}%")
    print(f"   (= both expressions give same answer)")
    print()
    print(f"📈 Original accuracy:   {eval_results['orig_accuracy']*100:.2f}%")
    print(f"📈 Augmented accuracy:  {eval_results['aug_accuracy']*100:.2f}%")
    print("=" * 70)
    
    # Show inconsistent examples
    if eval_results["samples"]:
        print("\n❌ Example Inconsistencies:")
        print("-" * 70)
        for sample in eval_results["samples"]:
            print(f"Type: {sample['type']}")
            print(f"  Orig:     {sample['orig']:30s} → {sample['pred_orig']:10s}")
            print(f"  Aug:      {sample['aug']:30s} → {sample['pred_aug']:10s}")
            print(f"  Expected: {sample['expected']}")
            print()
    
    print("=" * 70)
    print("✅ EC Probe Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

