"""
Arithmetic Model Evaluation Script
- Evaluates trained model on regular, OOD, and hard datasets
- Generates arithmetic expressions and computes accuracy scores
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch

import random

import os

import sys

# Add current directory to path for imports
# Handle both regular Python and Colab environments
try:
    # Regular Python environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Colab environment (__file__ is not defined)
    current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ========================
# Data Generators
# ========================


def gen_num(a: int = 1, b: int = 5) -> str:
    """Generate random number with specified digit range"""
    d = random.randint(a, b)
    return str(random.randint(10 ** (d - 1), 10 ** d - 1))


# --- REGULAR ---


def gen_CA() -> str:
    """Generate Chain Arithmetic expression"""
    n = random.randint(1, 3)
    e = gen_num()
    for _ in range(n):
        e += random.choice(["+", "-", "*", "//"]) + gen_num()
    if random.random() < 0.4:
        e = f"({e})"
    return e


def gen_LP() -> str:
    """Generate Left-Precedence expression"""
    a, b, c, d = gen_num(), gen_num(), gen_num(), gen_num()
    return random.choice([
        f"{a}+({b}*{c})",
        f"({a}+{b})*{c}",
        f"{a}//({b}+{c})",
        f"({a}*{b})//({c}+{d})",
        f"{a}+({b}//({c}+{d}))",
    ])


def gen_EC() -> str:
    """Generate Equivalence Class expression"""
    a, b, c, d = gen_num(), gen_num(), gen_num(), gen_num()
    return random.choice([
        f"{a}+{b}", f"{b}+{a}",
        f"({a}+{b})+{c}", f"{a}+({b}+{c})",
        f"{a}*({b}+{c})", f"{a}*{b}+{a}*{c}",
        f"({a}+{b})*({c}+{d})",
    ])


def gen_RC() -> str:
    """Generate Rational Combination expression"""
    a, b, c, d, e = gen_num(), gen_num(), gen_num(), gen_num(), gen_num()
    L = random.choice([f"{a}+{b}", f"{a}*{b}", f"{a}//({b}+{c})"])
    R = random.choice([f"{c}+{d}", f"{b}*{d}", f"({d}+{e})//{a}"])
    return f"{L}//{R}"


REG_FUNCS = {"CA": gen_CA, "LP": gen_LP, "EC": gen_EC, "RC": gen_RC}

REG_DIST = {"CA": 0.35, "LP": 0.20, "EC": 0.30, "RC": 0.15}


def sample_reg() -> str:
    """Sample regular category based on distribution"""
    r = random.random()
    acc = 0
    for k, v in REG_DIST.items():
        acc += v
        if r <= acc:
            return k
    return "CA"


# --- OOD (4~7 digits) ---


def gen_large() -> str:
    """Generate large number (4-7 digits)"""
    return gen_num(4, 7)


def gen_OOD() -> str:
    """Generate Out-of-Distribution expression"""
    a, b, c, d = gen_large(), gen_large(), gen_large(), gen_large()
    return random.choice([
        f"{a}+({b}*{c})",
        f"({a}//({b}+{c}))*{d}",
        f"({a}+{b})*({c}+{d})",
        f"{a}*({b}//({c}+{d}))",
    ])


# --- HARD EC ---


def gen_hEC() -> str:
    """Generate Hard Equivalence Class expression"""
    a, b, c, d = gen_num(), gen_num(), gen_num(), gen_num()
    return random.choice([
        f"(({a}+{b})+{c})+{d}",
        f"{a}+({b}+({c}+{d}))",
        f"{a}*({b}+({c}+{d}))",
        f"{a}*{c}+{a}*{d}+{b}*{c}+{b}*{d}",
    ])


# --- HARD LP ---


def gen_hLP() -> str:
    """Generate Hard Left-Precedence expression"""
    a, b, c, d, e = gen_num(), gen_num(), gen_num(), gen_num(), gen_num()
    return random.choice([
        f"{a}+({b}*({c}+{d}))-{e}",
        f"({a}*({b}+({c}*{d})))+{e}",
        f"(({a}+{b})*({c}+({d}*{e})))",
    ])


# --- HARD RC ---


def gen_hRC() -> str:
    """Generate Hard Rational Combination expression"""
    a, b, c, d, e, f = gen_num(), gen_num(), gen_num(), gen_num(), gen_num(), gen_num()
    L = random.choice([
        f"{a}+({b}*{c})",
        f"({a}+{b})*({c}+{d})",
        f"{a}//({b}+({c}*{d}))",
    ])
    R = random.choice([
        f"{d}+({e}*{f})",
        f"({d}+{e})*({a}+{b})",
        f"({e}+{f})//{c}",
    ])
    return f"{L}//{R}"


# ========================
# Dataset Generation
# ========================


def generate_datasets() -> Dict[str, List[Tuple[str, str]]]:
    """Generate all evaluation datasets"""
    print("Generating evaluation datasets...")
    
    # Regular dataset
    regular = []
    for _ in range(500):
        cat = sample_reg()
        regular.append((cat, REG_FUNCS[cat]()))
    
    # OOD dataset
    ood = [("OOD", gen_OOD()) for _ in range(200)]
    
    # Hard EC dataset
    hard_ec = [("EC_HARD", gen_hEC()) for _ in range(200)]
    
    # Hard LP dataset
    hard_lp = [("LP_HARD", gen_hLP()) for _ in range(200)]
    
    # Hard RC dataset
    hard_rc = [("RC_HARD", gen_hRC()) for _ in range(200)]
    
    print(f"Generated: regular=500, ood=200, hard_ec=200, hard_lp=200, hard_rc=200")
    
    return {
        "Regular": regular,
        "OOD": ood,
        "Hard_EC": hard_ec,
        "Hard_LP": hard_lp,
        "Hard_RC": hard_rc,
    }


# ========================
# Evaluation Functions
# ========================


def safe_eval(expr: str) -> str | None:
    """
    Safely evaluate arithmetic expression
    Returns None if evaluation fails
    Developer log: Improved handling for large numbers and division by zero
    """
    try:
        # Evaluate expression (Python's // is floor division)
        result = eval(expr)
        
        # Handle None, NaN, or infinity
        if result is None:
            return None
        if isinstance(result, float):
            if not (result == result):  # NaN check
                return None
            if abs(result) == float('inf'):
                return None
        
        # Convert to integer string
        int_result = int(result)
        return str(int_result)
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        return None
    except Exception:
        return None


def eval_dataset(name: str, data: List[Tuple[str, str]], model: Model, max_examples: int = 3, 
                 category_weights: Dict[str, float] | None = None) -> Dict[str, Any]:
    """
    Evaluate model on a dataset
    
    Args:
        name: Dataset name
        data: List of (category, expression) tuples
        model: Model instance
        max_examples: Number of example predictions to print
        category_weights: Optional dictionary of category weights for weighted scoring
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    total = 0
    correct = 0
    category_stats: Dict[str, Dict[str, int]] = {}
    
    example_count = 0
    examples = []
    all_expressions = []  # Store all expressions for Regular dataset examples
    
    for cat, expr in data:
        # Get ground truth
        gt = safe_eval(expr)
        if gt is None:
            continue
        
        # Store expression for examples
        all_expressions.append((cat, expr))
        
        # Get model prediction
        pred = model.predict(expr)
        
        # Update statistics
        total += 1
        is_correct = (pred == gt)
        if is_correct:
            correct += 1
        
        # Update category statistics
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1
        
        # Collect examples
        if example_count < max_examples:
            mark = "✓" if is_correct else "✗"
            examples.append({
                "mark": mark,
                "expr": expr,
                "gt": gt,
                "pred": pred,
                "cat": cat,
            })
            example_count += 1
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    # Calculate weighted score for Regular dataset
    weighted_score = None
    if category_weights and name == "Regular":
        weighted_score = 0.0
        total_weight = 0.0
        for cat, weight in category_weights.items():
            if cat in category_stats:
                cat_acc = (category_stats[cat]["correct"] / category_stats[cat]["total"]) if category_stats[cat]["total"] > 0 else 0.0
                weighted_score += cat_acc * weight
                total_weight += weight
        if total_weight > 0:
            weighted_score = weighted_score / total_weight * 100
    
    # Print results
    print(f"\nTotal samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    if weighted_score is not None:
        print(f"Weighted Score (1.0 max): {weighted_score:.2f}%")
    
    # Print category breakdown
    if category_stats:
        print(f"\nCategory breakdown:")
        for cat, stats in sorted(category_stats.items()):
            cat_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            weight_str = f" (weight: {category_weights.get(cat, 0.0):.2f})" if category_weights and cat in category_weights else ""
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_acc:.2f}%){weight_str}")
    
    # Print example expressions for Regular dataset
    if name == "Regular" and all_expressions:
        print(f"\nExample expressions by category:")
        shown_cats = set()
        for cat, expr in all_expressions[:20]:  # Show first 20 expressions
            if cat not in shown_cats:
                print(f"  [{cat}] {expr}")
                shown_cats.add(cat)
                if len(shown_cats) >= 4:  # Show one from each category
                    break
    
    # Print examples
    if examples:
        print(f"\nExample predictions (first {len(examples)}):")
        for ex in examples:
            cat_str = f" [{ex.get('cat', '')}]" if 'cat' in ex else ""
            print(f"  [{ex['mark']}]{cat_str} {ex['expr']}")
            print(f"      GT:   {ex['gt']}")
            print(f"      Pred: {ex['pred']}")
    
    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "weighted_score": weighted_score,
        "category_stats": category_stats,
    }


def main():
    """Main evaluation function"""
    print("="*60)
    print("Arithmetic Model Evaluation")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    try:
        model = Model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Generate datasets
    datasets = generate_datasets()
    
    # Evaluate each dataset
    results = []
    for name, data in datasets.items():
        # Use category weights for Regular dataset
        category_weights = REG_DIST if name == "Regular" else None
        result = eval_dataset(name, data, model, max_examples=3, category_weights=category_weights)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    total_samples = 0
    total_correct = 0
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        if result.get('weighted_score') is not None:
            print(f"  Weighted Score: {result['weighted_score']:.2f}% (1.0 max)")
        total_samples += result['total']
        total_correct += result['correct']
    
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

