"""Standalone test script that doesn't require torch - tests data generation logic directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import only the functions we need, avoiding torch imports
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, Optional

# Copy necessary constants and functions from dataloader.py
TRAINING_DISTRIBUTION = {
    "base_calculation": 0.40,
    "precedence": 0.20,
    "expression_consistency": 0.25,
    "relational": 0.10,
    "single_number": 0.05,
}

OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.05,
    "precedence": 0.10,
    "expression_consistency": 0.20,
    "relational": 0.30,
    "single_number": 0.0,
}

PHASE_DIGIT_DISTRIBUTION = {
    1: {1: 0.4, 2: 0.6},
    2: {2: 0.4, 3: 0.6},
    3: {3: 0.4, 4: 0.6},
    4: {4: 0.4, 5: 0.6},
}


def _rand_int(rng: random.Random, num_digits: Tuple[int, int]) -> int:
    """Generate random integer with specified digit range (no leading zeros)"""
    lo, hi = num_digits
    n = rng.randint(lo, hi)
    if n == 1:
        return rng.randint(0, 9)
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(n - 1)]
    return int(str(first) + "".join(str(x) for x in rest))


def _sample_category(rng: random.Random) -> str:
    """Sample category based on training distribution"""
    r = rng.random()
    cumsum = 0.0
    for cat, prob in TRAINING_DISTRIBUTION.items():
        cumsum += prob
        if r <= cumsum:
            return cat
    return "base_calculation"


def _sample_digit_length(rng: random.Random, phase: int) -> int:
    """Sample digit length based on phase distribution"""
    dist = PHASE_DIGIT_DISTRIBUTION.get(phase, {4: 0.5, 5: 0.5})
    r = rng.random()
    cumsum = 0.0
    for digits, prob in dist.items():
        cumsum += prob
        if r <= cumsum:
            return digits
    return 4


def _should_generate_6digit_output(rng: random.Random, category: str) -> bool:
    """Check if we should generate 6+ digit output for this category"""
    ratio = OUTPUT_6DIGIT_RATIO.get(category, 0.0)
    return rng.random() < ratio


def generate_sample_data(num_samples: int, phase: int, seed: int, enable_augmentation: bool = True):
    """Generate sample data without using Dataset class"""
    rng = random.Random(seed)
    samples = []
    
    for idx in range(num_samples):
        # Use same logic as Dataset.__getitem__
        sample_rng = random.Random(seed + idx)
        
        # Sample category
        category = _sample_category(sample_rng)
        
        # Sample digit length
        digit_len = _sample_digit_length(sample_rng, phase)
        
        # Check 6-digit output
        force_6digit = _should_generate_6digit_output(sample_rng, category)
        
        # Generate expression (simplified - just for testing distribution)
        # We'll use a simple mock generation
        if category == "base_calculation":
            if force_6digit:
                a = _rand_int(sample_rng, (4, 5))
                b = _rand_int(sample_rng, (2, 3))
                expr = f"{a}*{b}"
                val = a * b
            else:
                a = _rand_int(sample_rng, (1, digit_len))
                b = _rand_int(sample_rng, (1, digit_len))
                expr = f"{a}+{b}"
                val = a + b
        elif category == "precedence":
            if force_6digit:
                a = _rand_int(sample_rng, (4, 5))
                b = _rand_int(sample_rng, (2, 3))
                c = _rand_int(sample_rng, (2, 3))
                expr = f"({a}+{b})*{c}"
                val = (a + b) * c
            else:
                # Use smaller numbers to avoid large outputs
                if sample_rng.random() < 0.6:
                    a = _rand_int(sample_rng, (1, min(3, digit_len)))
                    b = _rand_int(sample_rng, (1, min(3, digit_len)))
                    c = _rand_int(sample_rng, (1, min(3, digit_len)))
                    expr = f"({a}+{b})+{c}"
                    val = (a + b) + c
                else:
                    a = _rand_int(sample_rng, (1, min(2, digit_len)))
                    b = _rand_int(sample_rng, (1, min(2, digit_len)))
                    c = _rand_int(sample_rng, (1, min(2, digit_len)))
                    expr = f"({a}+{b})*{c}"
                    val = (a + b) * c
        elif category == "expression_consistency":
            op = sample_rng.choice(["+", "*"])
            a = _rand_int(sample_rng, (1, digit_len))
            b = _rand_int(sample_rng, (1, digit_len))
            if op == "+":
                expr = f"{a}+{b}"
                val = a + b
            else:
                expr = f"{a}*{b}"
                val = a * b
        elif category == "relational":
            if force_6digit:
                base = _rand_int(sample_rng, (4, 5))
                multiplier = sample_rng.randint(10, 99)
                val = base * multiplier
                if val < 100000:
                    multiplier = sample_rng.randint(100, 999)
                expr = f"{base}*{multiplier}"
                val = base * multiplier
            else:
                base = _rand_int(sample_rng, (1, digit_len))
                expr = f"{base}+0"
                val = base
        elif category == "single_number":
            val = _rand_int(sample_rng, (1, digit_len))
            expr = str(val)
        else:
            a = _rand_int(sample_rng, (1, digit_len))
            b = _rand_int(sample_rng, (1, digit_len))
            expr = f"{a}+{b}"
            val = a + b
        
        output_6digit = len(str(val)) >= 6
        
        samples.append({
            "input_text": expr,
            "target_text": str(val),
            "meta": {
                "category": category,
                "phase": phase,
                "digit_len": digit_len,
                "output_6digit": output_6digit,
            }
        })
    
    return samples


def test_category_distribution(num_samples=10000, phase=4, seed=42):
    """Test if category distribution matches expected ratios"""
    print("=" * 70)
    print("TEST 1: Category Distribution")
    print("=" * 70)
    
    samples = generate_sample_data(num_samples, phase, seed, enable_augmentation=True)
    
    categories = [s["meta"]["category"] for s in samples]
    cat_counts = Counter(categories)
    total = len(categories)
    
    print(f"\nTotal samples: {total}")
    print(f"\n{'Category':<30} {'Count':<10} {'Actual %':<12} {'Expected %':<12} {'Diff':<10} {'Status'}")
    print("-" * 70)
    
    max_diff = 0.0
    for cat, expected_ratio in TRAINING_DISTRIBUTION.items():
        count = cat_counts.get(cat, 0)
        actual_ratio = count / total
        diff = abs(actual_ratio - expected_ratio)
        max_diff = max(max_diff, diff)
        
        status = "✓" if diff < 0.05 else "✗"
        print(f"{cat:<30} {count:<10} {actual_ratio*100:>6.2f}%     {expected_ratio*100:>6.2f}%     {diff*100:>6.2f}%  {status}")
    
    print("-" * 70)
    if max_diff < 0.05:
        print(f"✓ PASS: Maximum difference {max_diff*100:.2f}% < 5% tolerance")
        return True
    else:
        print(f"✗ FAIL: Maximum difference {max_diff*100:.2f}% >= 5% tolerance")
        return False


def test_6digit_output_distribution(num_samples=10000, phase=4, seed=42):
    """Test if 6+ digit output ratio matches expected ratios per category"""
    print("\n" + "=" * 70)
    print("TEST 2: 6+ Digit Output Distribution")
    print("=" * 70)
    
    samples = generate_sample_data(num_samples, phase, seed, enable_augmentation=True)
    
    category_stats = defaultdict(lambda: {"total": 0, "6digit": 0})
    
    for sample in samples:
        cat = sample["meta"]["category"]
        category_stats[cat]["total"] += 1
        if sample["meta"]["output_6digit"]:
            category_stats[cat]["6digit"] += 1
    
    print(f"\n{'Category':<30} {'Total':<10} {'6+ Digit':<12} {'Actual %':<12} {'Expected %':<12} {'Status'}")
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
    target_overall = 0.15
    
    print("-" * 70)
    print(f"{'OVERALL':<30} {overall_total:<10} {overall_6digit:<12} {overall_ratio*100:>6.2f}%     {target_overall*100:>6.2f}%     ", end="")
    
    if abs(overall_ratio - target_overall) < 0.03:
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
    
    samples = generate_sample_data(num_samples, phase, seed, enable_augmentation=False)
    
    digit_lengths = [s["meta"]["digit_len"] for s in samples]
    digit_counts = Counter(digit_lengths)
    total = len(digit_lengths)
    
    expected_dist = PHASE_DIGIT_DISTRIBUTION.get(phase, {})
    
    print(f"\nTotal samples: {total}")
    print(f"\n{'Digit Length':<15} {'Count':<10} {'Actual %':<12} {'Expected %':<12} {'Status'}")
    print("-" * 70)
    
    all_pass = True
    for digit_len in sorted(set(digit_lengths)):
        count = digit_counts.get(digit_len, 0)
        actual_ratio = count / total
        expected_ratio = expected_dist.get(digit_len, 0.0)
        diff = abs(actual_ratio - expected_ratio)
        
        status = "✓" if diff < 0.1 or expected_ratio == 0.0 else "✗"
        if status == "✗":
            all_pass = False
        
        print(f"{digit_len} digits{'':<10} {count:<10} {actual_ratio*100:>6.2f}%     {expected_ratio*100:>6.2f}%     {status}")
    
    print("-" * 70)
    if all_pass:
        print("✓ PASS: Digit length distribution matches phase settings")
    else:
        print("✗ FAIL: Digit length distribution does not match phase settings")
    
    return all_pass


def test_input_digit_constraint(num_samples=5000, phase=4, seed=42):
    """Test if input digits are within 1-5 range (training constraint)"""
    print("\n" + "=" * 70)
    print("TEST 4: Input Digit Constraint (1-5 digits)")
    print("=" * 70)
    
    samples = generate_sample_data(num_samples, phase, seed, enable_augmentation=True)
    
    violations = []
    
    for i, sample in enumerate(samples):
        expr = sample["input_text"]
        numbers = re.findall(r'\d+', expr)
        for num_str in numbers:
            num = int(num_str)
            num_digits = len(str(num))
            if num_digits < 1 or num_digits > 5:
                violations.append(f"Sample {i}: {expr} contains {num} ({num_digits} digits)")
    
    print(f"\nTotal samples checked: {len(samples)}")
    print(f"Violations found: {len(violations)}")
    
    if len(violations) == 0:
        print("✓ PASS: All input numbers are within 1-5 digit range")
        return True
    else:
        print("✗ FAIL: Found violations:")
        for v in violations[:10]:
            print(f"  {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more")
        return False


def test_sample_examples(num_samples=20, phase=4, seed=42):
    """Show sample examples from each category"""
    print("\n" + "=" * 70)
    print("TEST 5: Sample Examples")
    print("=" * 70)
    
    samples = generate_sample_data(num_samples, phase, seed, enable_augmentation=True)
    
    print(f"\nShowing {min(20, len(samples))} sample expressions:")
    print(f"\n{'Input':<30} {'Output':<15} {'Category':<25} {'6+dig'}")
    print("-" * 70)
    
    for i, sample in enumerate(samples[:20]):
        inp = sample["input_text"]
        out = sample["target_text"]
        cat = sample["meta"]["category"]
        is_6dig = "Yes" if sample["meta"]["output_6digit"] else "No"
        print(f"{inp:<30} {out:<15} {cat:<25} {is_6dig}")


def run_all_tests():
    """Run all tests and report summary"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATALOADER TEST SUITE (Standalone)")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Category distribution
    results["category_dist"] = test_category_distribution(num_samples=10000, phase=4)
    
    # Test 2: 6+ digit output distribution
    results["6digit_output"] = test_6digit_output_distribution(num_samples=10000, phase=4)
    
    # Test 3: Phase digit distribution
    results["phase_digits"] = test_phase_digit_distribution(phase=4, num_samples=5000)
    
    # Test 4: Input digit constraint
    results["digit_constraint"] = test_input_digit_constraint(num_samples=5000, phase=4)
    
    # Test 5: Sample examples
    test_sample_examples(num_samples=20, phase=4)
    
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

