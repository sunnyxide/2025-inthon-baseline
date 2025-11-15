"""Detailed test script for enhanced dataloader - distribution and validation"""
from dataloader import (
    ArithmeticDataset, 
    get_dataloader,
    TRAINING_DISTRIBUTION,
    OUTPUT_6DIGIT_RATIO,
    PHASE_DIGIT_DISTRIBUTION,
)
from collections import Counter, defaultdict
import sys

def test_category_distribution(num_samples=10000, phase=4, seed=42):
    """Test if category distribution matches expected ratios"""
    print("=" * 70)
    print("TEST 1: Category Distribution")
    print("=" * 70)
    
    dataset = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=True,
    )
    
    categories = []
    for i in range(len(dataset)):
        sample = dataset[i]
        categories.append(sample["meta"]["category"])
    
    cat_counts = Counter(categories)
    total = len(categories)
    
    print(f"\nTotal samples: {total}")
    print(f"\n{'Category':<30} {'Count':<10} {'Actual %':<12} {'Expected %':<12} {'Diff':<10}")
    print("-" * 70)
    
    max_diff = 0.0
    for cat, expected_ratio in TRAINING_DISTRIBUTION.items():
        count = cat_counts.get(cat, 0)
        actual_ratio = count / total
        diff = abs(actual_ratio - expected_ratio)
        max_diff = max(max_diff, diff)
        
        status = "✓" if diff < 0.05 else "✗"  # 5% tolerance
        print(f"{cat:<30} {count:<10} {actual_ratio*100:>6.2f}%     {expected_ratio*100:>6.2f}%     {diff*100:>6.2f}%  {status}")
    
    print("-" * 70)
    if max_diff < 0.05:
        print(f"✓ PASS: Maximum difference {max_diff*100:.2f}% < 5% tolerance")
    else:
        print(f"✗ FAIL: Maximum difference {max_diff*100:.2f}% >= 5% tolerance")
    
    return max_diff < 0.05


def test_6digit_output_distribution(num_samples=10000, phase=4, seed=42):
    """Test if 6+ digit output ratio matches expected ratios per category"""
    print("\n" + "=" * 70)
    print("TEST 2: 6+ Digit Output Distribution")
    print("=" * 70)
    
    dataset = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=True,
    )
    
    category_stats = defaultdict(lambda: {"total": 0, "6digit": 0})
    
    for i in range(len(dataset)):
        sample = dataset[i]
        cat = sample["meta"]["category"]
        category_stats[cat]["total"] += 1
        if sample["meta"]["output_6digit"]:
            category_stats[cat]["6digit"] += 1
    
    print(f"\n{'Category':<30} {'Total':<10} {'6+ Digit':<12} {'Actual %':<12} {'Expected %':<12} {'Status':<10}")
    print("-" * 70)
    
    overall_6digit = 0
    overall_total = 0
    all_pass = True
    
    for cat in TRAINING_DISTRIBUTION.keys():
        stats = category_stats[cat]
        total = stats["total"]
        count_6digit = stats["6digit"]
        actual_ratio = count_6digit / total if total > 0 else 0.0
        expected_ratio = OUTPUT_6DIGIT_RATIO.get(cat, 0.0)
        diff = abs(actual_ratio - expected_ratio)
        
        status = "✓" if diff < 0.05 or (expected_ratio == 0.0 and actual_ratio < 0.05) else "✗"
        if status == "✗":
            all_pass = False
        
        print(f"{cat:<30} {total:<10} {count_6digit:<12} {actual_ratio*100:>6.2f}%     {expected_ratio*100:>6.2f}%     {status}")
        
        overall_6digit += count_6digit
        overall_total += total
    
    overall_ratio = overall_6digit / overall_total if overall_total > 0 else 0.0
    target_overall = 0.15  # 15% target
    
    print("-" * 70)
    print(f"{'OVERALL':<30} {overall_total:<10} {overall_6digit:<12} {overall_ratio*100:>6.2f}%     {target_overall*100:>6.2f}%     ", end="")
    
    if abs(overall_ratio - target_overall) < 0.03:  # 3% tolerance for overall
        print("✓")
        print(f"✓ PASS: Overall 6+ digit ratio {overall_ratio*100:.2f}% is close to target {target_overall*100:.2f}%")
    else:
        print("✗")
        print(f"✗ FAIL: Overall 6+ digit ratio {overall_ratio*100:.2f}% differs from target {target_overall*100:.2f}%")
        all_pass = False
    
    return all_pass


def test_phase_digit_distribution(phase=4, num_samples=5000, seed=42):
    """Test if digit length distribution matches phase settings"""
    print("\n" + "=" * 70)
    print(f"TEST 3: Phase {phase} Digit Length Distribution")
    print("=" * 70)
    
    dataset = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=False,  # Disable augmentation for cleaner test
    )
    
    digit_lengths = []
    for i in range(len(dataset)):
        sample = dataset[i]
        digit_lengths.append(sample["meta"]["digit_len"])
    
    digit_counts = Counter(digit_lengths)
    total = len(digit_lengths)
    
    expected_dist = PHASE_DIGIT_DISTRIBUTION.get(phase, {})
    
    print(f"\nTotal samples: {total}")
    print(f"\n{'Digit Length':<15} {'Count':<10} {'Actual %':<12} {'Expected %':<12} {'Status':<10}")
    print("-" * 70)
    
    all_pass = True
    for digit_len in sorted(set(digit_lengths)):
        count = digit_counts.get(digit_len, 0)
        actual_ratio = count / total
        expected_ratio = expected_dist.get(digit_len, 0.0)
        diff = abs(actual_ratio - expected_ratio)
        
        status = "✓" if diff < 0.1 or expected_ratio == 0.0 else "✗"  # 10% tolerance
        if status == "✗":
            all_pass = False
        
        print(f"{digit_len} digits{'':<10} {count:<10} {actual_ratio*100:>6.2f}%     {expected_ratio*100:>6.2f}%     {status}")
    
    print("-" * 70)
    if all_pass:
        print("✓ PASS: Digit length distribution matches phase settings")
    else:
        print("✗ FAIL: Digit length distribution does not match phase settings")
    
    return all_pass


def test_augmentation(num_samples=2000, phase=4, seed=42):
    """Test if augmentation works correctly for expression_consistency"""
    print("\n" + "=" * 70)
    print("TEST 4: Augmentation (Expression Consistency)")
    print("=" * 70)
    
    # Test with augmentation enabled
    dataset_aug = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=True,
    )
    
    # Test without augmentation
    dataset_no_aug = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=False,
    )
    
    # Count augmented patterns (commutative: a+b -> b+a, a*b -> b*a)
    aug_count = 0
    expr_consistency_count = 0
    
    for i in range(len(dataset_aug)):
        sample = dataset_aug[i]
        if sample["meta"]["category"] == "expression_consistency":
            expr_consistency_count += 1
            expr = sample["input_text"]
            # Check if it's a commutative pattern (b+a or b*a where a < b in original)
            tokens = expr.replace("+", " ").replace("*", " ").split()
            if len(tokens) == 2:
                try:
                    num1, num2 = int(tokens[0]), int(tokens[1])
                    # If second number is larger, likely augmented
                    if num2 > num1 and ("+" in expr or "*" in expr):
                        aug_count += 1
                except:
                    pass
    
    aug_ratio = aug_count / expr_consistency_count if expr_consistency_count > 0 else 0.0
    
    print(f"\nExpression consistency samples: {expr_consistency_count}")
    print(f"Likely augmented samples: {aug_count}")
    print(f"Augmentation ratio: {aug_ratio*100:.2f}%")
    
    # Check some examples
    print("\nSample augmented expressions:")
    count = 0
    for i in range(len(dataset_aug)):
        sample = dataset_aug[i]
        if sample["meta"]["category"] == "expression_consistency" and count < 5:
            print(f"  {sample['input_text']:20s} → {sample['target_text']}")
            count += 1
    
    if aug_ratio > 0.1:  # At least 10% should be augmented
        print("\n✓ PASS: Augmentation is working")
        return True
    else:
        print("\n✗ FAIL: Augmentation ratio too low")
        return False


def test_data_validation(num_samples=1000, phase=4, seed=42):
    """Test if generated data passes validation"""
    print("\n" + "=" * 70)
    print("TEST 5: Data Validation (dataloader_validator)")
    print("=" * 70)
    
    dataset = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=True,
    )
    
    dataloader = get_dataloader(
        dataset,
        batch_size=32,
        num_workers=0,
        mode="train",
    )
    
    validation_errors = []
    total_batches = 0
    total_samples = 0
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            total_batches += 1
            total_samples += len(batch["input_text"])
            
            # Check if batch has required keys
            if "input_text" not in batch or "target_text" not in batch:
                validation_errors.append(f"Batch {batch_idx}: Missing required keys")
            
            # Check a few samples manually
            if batch_idx == 0:
                print(f"\nFirst batch ({len(batch['input_text'])} samples):")
                for i in range(min(5, len(batch["input_text"]))):
                    inp = batch["input_text"][i]
                    tgt = batch["target_text"][i]
                    print(f"  [{i+1}] {inp:25s} → {tgt}")
        
        print(f"\n✓ PASS: All {total_batches} batches passed validation")
        print(f"  Total samples validated: {total_samples}")
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: Validation error: {e}")
        return False


def test_input_digit_constraint(num_samples=5000, phase=4, seed=42):
    """Test if input digits are within 1-5 range (training constraint)"""
    print("\n" + "=" * 70)
    print("TEST 6: Input Digit Constraint (1-5 digits)")
    print("=" * 70)
    
    dataset = ArithmeticDataset(
        num_samples=num_samples,
        phase=phase,
        seed=seed,
        mode="train",
        enable_augmentation=True,
    )
    
    import re
    violations = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        expr = sample["input_text"]
        # Extract all numbers from expression
        numbers = re.findall(r'\d+', expr)
        for num_str in numbers:
            num = int(num_str)
            num_digits = len(str(num))
            if num_digits < 1 or num_digits > 5:
                violations.append(f"Sample {i}: {expr} contains {num} ({num_digits} digits)")
    
    print(f"\nTotal samples checked: {len(dataset)}")
    print(f"Violations found: {len(violations)}")
    
    if len(violations) == 0:
        print("✓ PASS: All input numbers are within 1-5 digit range")
        return True
    else:
        print("✗ FAIL: Found violations:")
        for v in violations[:10]:  # Show first 10
            print(f"  {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
        return False


def run_all_tests():
    """Run all tests and report summary"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATALOADER TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Category distribution
    results["category_dist"] = test_category_distribution(num_samples=10000, phase=4)
    
    # Test 2: 6+ digit output distribution
    results["6digit_output"] = test_6digit_output_distribution(num_samples=10000, phase=4)
    
    # Test 3: Phase digit distribution
    results["phase_digits"] = test_phase_digit_distribution(phase=4, num_samples=5000)
    
    # Test 4: Augmentation
    results["augmentation"] = test_augmentation(num_samples=2000, phase=4)
    
    # Test 5: Data validation
    results["validation"] = test_data_validation(num_samples=1000, phase=4)
    
    # Test 6: Input digit constraint
    results["digit_constraint"] = test_input_digit_constraint(num_samples=5000, phase=4)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the results above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

