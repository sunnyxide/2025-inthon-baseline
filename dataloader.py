"""Arithmetic data generator with category-aware curriculum and comprehensive augmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import random
import re
from functools import partial

from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation

# ---------------------------------------------------------------------------
# Distribution targets (literature-backed v0 plan)
# ---------------------------------------------------------------------------

TRAINING_DISTRIBUTION = {
    "base_calculation": 0.25,        # 40% → 25% (CA 유지하면서 축소)
    "precedence": 0.15,              # 20% → 15% (LP 유지하면서 축소)
    "expression_consistency": 0.45,  # 25% → 45% ⬆️ EC 집중! (안정적 상한선)
    "relational": 0.10,              # 10% 유지 (RC 이미 84%로 우수)
    "single_number": 0.05,           # 5% baseline 유지
}

PHASE_DIGIT_DISTRIBUTION = {
    1: {1: 0.4, 2: 0.6},
    2: {2: 0.4, 3: 0.6},
    3: {3: 0.4, 4: 0.6},
    4: {4: 0.4, 5: 0.6},
}

OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.15,        # 0.05 → 0.15 (큰수 연산 강화)
    "precedence": 0.20,              # 0.10 → 0.20 (큰수 연산 강화)
    "expression_consistency": 0.35,  # 0.30 → 0.35 (큰수 EC 강화)
    "relational": 0.40,              # 0.30 → 0.40 (큰수 관계성 강화)
    "single_number": 0.00,
}

PHASE_AUGMENTATION_PROB = {
    1: 0.15,  # 10% → 15% (augmentation 증가)
    2: 0.30,  # 20% → 30%
    3: 0.45,  # 35% → 45%
    4: 0.35,  # 25% → 35%
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _rand_int(rng: random.Random, num_digits: Tuple[int, int]) -> int:
    """Generate random integer with specified digit length."""
    lo, hi = num_digits
    digits = rng.randint(lo, hi)
    if digits == 1:
        return rng.randint(0, 9)
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(digits - 1)]
    return int(str(first) + "".join(str(d) for d in rest))


def _sample_category(rng: random.Random) -> str:
    """Sample category based on distribution."""
    r = rng.random()
    cumsum = 0.0
    for cat, prob in TRAINING_DISTRIBUTION.items():
        cumsum += prob
        if r <= cumsum:
            return cat
    return "base_calculation"


def _sample_digit_length(rng: random.Random, phase: int) -> int:
    """Sample digit length based on phase."""
    dist = PHASE_DIGIT_DISTRIBUTION.get(phase, PHASE_DIGIT_DISTRIBUTION[4])
    r = rng.random()
    cumsum = 0.0
    for digits, prob in dist.items():
        cumsum += prob
        if r <= cumsum:
            return digits
    return max(dist, key=dist.get)


def _should_force_large_output(rng: random.Random, category: str) -> bool:
    """Check if we should force large output (6+ digits)."""
    return rng.random() < OUTPUT_6DIGIT_RATIO.get(category, 0.0)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------


def _apply_distributive_law(
    expr: str, val: int, rng: random.Random
) -> Optional[Tuple[str, int]]:
    """Apply distributive law: a*(b+c) → a*b+a*c."""
    # Match a*(b±c) or (b±c)*a
    match = re.match(r'^(\d+)\*\((\d+)([\+\-])(\d+)\)$', expr)
    if not match:
        match = re.match(r'^\((\d+)([\+\-])(\d+)\)\*(\d+)$', expr)
        if match:
            b, op, c, a = match.groups()
            a, b, c = int(a), int(b), int(c)
        else:
            return None
    else:
        a, b, op, c = match.groups()
        a, b, c = int(a), int(b), int(c)
    
    # Apply distributive law
    if op == '+':
        new_expr = f"{a}*{b}+{a}*{c}"
        new_val = a * b + a * c
    else:  # op == '-'
        new_expr = f"{a}*{b}-{a}*{c}"
        new_val = a * b - a * c
    
    if new_val == val:
        return new_expr, new_val
    return None


def _apply_identity_augmentation(
    expr: str, val: int, rng: random.Random
) -> Optional[Tuple[str, int]]:
    """Apply identity augmentation: expr → expr+0, expr-0, expr*1, 0+expr."""
    augmentations = [
        (f"{expr}+0", val),
        (f"0+{expr}", val),
        (f"{expr}-0", val),
        (f"{expr}*1", val),
        (f"1*{expr}", val),
    ]
    return rng.choice(augmentations)


def _apply_commutative_law(
    expr: str, val: int, rng: random.Random
) -> Optional[Tuple[str, int]]:
    """Apply commutative law for simple expressions: a+b → b+a, a*b → b*a."""
    # Match simple binary expressions without parentheses
    tokens = re.findall(r'\d+|[\+\*]', expr)
    if len(tokens) == 3 and '(' not in expr:
        lhs, op, rhs = tokens
        if op in ['+', '*']:
            try:
                left = int(lhs)
                right = int(rhs)
                swapped = f"{right}{op}{left}"
                new_val = right + left if op == '+' else right * left
                if new_val == val:
                    return swapped, new_val
            except ValueError:
                return None
    return None


def _apply_associative_law(
    expr: str, val: int, rng: random.Random
) -> Optional[Tuple[str, int]]:
    """Apply associative law: (a+b)+c → a+(b+c), (a*b)*c → a*(b*c)."""
    match = re.match(r'^\((\d+)([\+\*])(\d+)\)([\+\*])(\d+)$', expr)
    if not match:
        return None
    a_str, op1, b_str, op2, c_str = match.groups()
    if op1 != op2:
        return None
    if op1 not in ['+', '*']:
        return None
    a, b, c = int(a_str), int(b_str), int(c_str)
    if op1 == '+':
        new_expr = f"{a}+({b}+{c})"
        new_val = a + (b + c)
    else:
        new_expr = f"{a}*({b}*{c})"
        new_val = a * (b * c)
    if new_val == val:
        return new_expr, new_val
    return None


def _safe_augment_expression(
    expr: str,
    val: int,
    rng: random.Random,
) -> Optional[Tuple[str, int]]:
    """
    Apply various augmentation strategies with proper probability distribution.
    Developer log: Comprehensive augmentation including distributive, commutative, 
    associative laws and identity operations.
    """
    # List of augmentation strategies
    strategies = [
        _apply_commutative_law,      # a+b → b+a, a*b → b*a
        _apply_associative_law,      # (a+b)+c → a+(b+c)
        _apply_distributive_law,     # a*(b+c) → a*b+a*c
        _apply_identity_augmentation, # expr → expr+0, expr*1
    ]
    
    # Try strategies in random order
    rng.shuffle(strategies)
    for strategy in strategies:
        result = strategy(expr, val, rng)
        if result is not None:
            return result
    
    return None


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------


def _gen_base_calculation(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """Generate basic calculation expressions with proper operator precedence."""
    if force_large:
        if rng.random() < 0.7:
            a = _rand_int(rng, (4, 5))
            b = _rand_int(rng, (2, 3))
            return f"{a}*{b}", a * b
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (4, 5))
        return f"{a}+{b}", a + b

    def make_number() -> Tuple[str, int]:
        v = _rand_int(rng, (1, digit_len))
        return str(v), v

    def make_term(depth: int) -> Tuple[str, int]:
        if depth == 0 or rng.random() < 0.5:
            return make_number()
        op = rng.choice(["*", "//"])
        left_expr, left_val = make_term(rng.randint(0, depth - 1))
        right_expr, right_val = make_number()
        if op == "//" and right_val == 0:
            right_val = rng.randint(1, 9)
            right_expr = str(right_val)
        val = left_val // right_val if op == "//" else left_val * right_val
        return f"{left_expr}{op}{right_expr}", val

    def make_expr(depth: int) -> Tuple[str, int]:
        if depth == 0 or rng.random() < 0.5:
            return make_term(depth)
        op = rng.choice(["+", "-"])
        left_expr, left_val = make_expr(rng.randint(0, depth - 1))
        right_expr, right_val = make_term(rng.randint(0, depth - 1))
        
        # Prevent negative results - ensure left >= right for subtraction
        if op == "-":
            if left_val < right_val:
                left_expr, right_expr = right_expr, left_expr
                left_val, right_val = right_val, left_val
            # Additional check for nested expressions
            if left_val < right_val:
                op = "+"  # Fallback to addition
        
        val = left_val + right_val if op == "+" else left_val - right_val
        return f"{left_expr}{op}{right_expr}", val

    depth = rng.randint(1, 2)
    expr, value = make_expr(depth)
    return expr, value


def _gen_precedence(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate precedence-focused expressions with diverse patterns.
    Developer log: Added more diverse precedence patterns including (a*b)+c.
    """
    if force_large:
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        c = _rand_int(rng, (2, 3))
        if rng.random() < 0.5:
            return f"({a}+{b})*{c}", (a + b) * c
        return f"{a}*({b}+{c})", a * (b + c)

    # More diverse precedence patterns
    pattern_choice = rng.random()
    
    if pattern_choice < 0.3:
        # Addition/subtraction with parentheses
        a = _rand_int(rng, (1, min(3, digit_len)))
        b = _rand_int(rng, (1, min(3, digit_len)))
        c = _rand_int(rng, (1, min(3, digit_len)))
        if rng.random() < 0.5:
            val = (a + b) - c
            if val < 0:
                val = (a + b) + c
                return f"({a}+{b})+{c}", val
            return f"({a}+{b})-{c}", val
        return f"{a}+({b}+{c})", a + (b + c)
    
    elif pattern_choice < 0.6:
        # Multiplication with addition/subtraction in parentheses
        a = _rand_int(rng, (1, min(2, digit_len)))
        b = _rand_int(rng, (1, min(2, digit_len)))
        c = _rand_int(rng, (1, min(2, digit_len)))
        if rng.random() < 0.5:
            return f"({a}+{b})*{c}", (a + b) * c
        return f"{a}*({b}+{c})", a * (b + c)
    
    else:
        # Multiplication precedence: (a*b)+c or a+(b*c)
        a = _rand_int(rng, (1, min(2, digit_len)))
        b = _rand_int(rng, (1, min(2, digit_len)))
        c = _rand_int(rng, (1, min(2, digit_len)))
        if rng.random() < 0.5:
            return f"({a}*{b})+{c}", (a * b) + c
        return f"{a}+({b}*{c})", a + (b * c)


def _gen_expression_consistency_base(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate expressions for consistency testing.
    EC 강화: 교환법칙 패턴에 집중 (A+B vs B+A, A*B vs B*A)
    """
    if force_large:
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        op = rng.choice(["+", "*"])
        # 50/50으로 순서 랜덤화
        if rng.random() < 0.5:
            return f"{a}{op}{b}", (a + b if op == "+" else a * b)
        return f"{b}{op}{a}", (a + b if op == "+" else a * b)

    # 75% 확률로 단순 이항 연산 (교환법칙 집중) - EC 강화: 60% → 75%
    if rng.random() < 0.75:
        op = rng.choice(["+", "*"])
        a = _rand_int(rng, (1, digit_len))
        b = _rand_int(rng, (1, digit_len))
        # 50/50으로 순서 랜덤화 → A op B와 B op A를 고루 학습
        if rng.random() < 0.5:
            expr = f"{a}{op}{b}"
        else:
            expr = f"{b}{op}{a}"
        val = a + b if op == "+" else a * b
        return expr, val

    # 25% 확률로 결합법칙 패턴 ((A+B)+C vs A+(B+C))
    op = rng.choice(["+", "*"])
    a = _rand_int(rng, (1, digit_len))
    b = _rand_int(rng, (1, digit_len))
    c = _rand_int(rng, (1, digit_len))
    
    # 50/50으로 괄호 위치 변경 (결합법칙)
    if rng.random() < 0.5:
        expr = f"({a}{op}{b}){op}{c}"
    else:
        expr = f"{a}{op}({b}{op}{c})"
    
    if op == "+":
        val = a + b + c
    else:
        val = a * b * c
    return expr, val


def _gen_relational(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate relational patterns focusing on identities.
    Developer log: Removed single number pattern to avoid overlap with single_number category.
    """
    if force_large:
        base = _rand_int(rng, (4, 5))
        mul = rng.randint(50, 999)
        return f"{base}*{mul}", base * mul

    base = _rand_int(rng, (1, digit_len))
    # Removed (f"{base}", base) to avoid overlap
    patterns = [
        (f"{base}+0", base),
        (f"0+{base}", base),
        (f"{base}*1", base),
        (f"1*{base}", base),
        (f"{base}*0", 0),
        (f"0*{base}", 0),
        (f"{base}+1", base + 1),
        (f"{base}+2", base + 2),
    ]
    return rng.choice(patterns)


def _gen_single_number(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """Generate single number (for curriculum completeness)."""
    val = _rand_int(rng, (5, 5)) if force_large else _rand_int(rng, (1, digit_len))
    text = str(val)
    return text, val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ArithmeticDataset(Dataset):
    """
    Arithmetic dataset with category-aware curriculum and comprehensive augmentation.
    Developer log: Supports both phase and max_depth parameters for backward compatibility.
    """
    
    def __init__(
        self,
        num_samples: int,
        phase: Optional[int] = None,
        seed: int = 42,
        mode: str = "train",
        enable_augmentation: bool = True,
        num_digits: Optional[Tuple[int, int]] = None,
        max_depth: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.mode = mode
        self.enable_augmentation = enable_augmentation

        if phase is None:
            if max_depth is not None:
                # max_depth를 직접 phase로 변환 (더 직관적)
                # max_depth=2 → phase=2 (2-3자리 숫자 생성)
                phase = min(max(1, max_depth), 4)
            elif num_digits is not None:
                # num_digits를 phase로 변환
                _, max_digits = num_digits
                if max_digits <= 2:
                    phase = 1
                elif max_digits <= 3:
                    phase = 2
                elif max_digits <= 4:
                    phase = 3
                else:
                    phase = 4
            else:
                phase = 4
        self.phase = phase

        self._category_cumsum: List[Tuple[float, str]] = []
        cumulative = 0.0
        for cat, prob in TRAINING_DISTRIBUTION.items():
            cumulative += prob
            self._category_cumsum.append((cumulative, cat))

    def __len__(self) -> int:
        return self.num_samples

    def _sample_category(self, rng: random.Random) -> str:
        r = rng.random()
        for threshold, cat in self._category_cumsum:
            if r <= threshold:
                return cat
        return self._category_cumsum[-1][1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        category = self._sample_category(rng)
        digit_len = _sample_digit_length(rng, self.phase)
        force_large = _should_force_large_output(rng, category)

        # Generate base expression
        if category == "base_calculation":
            expr, val = _gen_base_calculation(rng, digit_len, force_large)
        elif category == "precedence":
            expr, val = _gen_precedence(rng, digit_len, force_large)
        elif category == "expression_consistency":
            expr, val = _gen_expression_consistency_base(rng, digit_len, force_large)
        elif category == "relational":
            expr, val = _gen_relational(rng, digit_len, force_large)
        elif category == "single_number":
            expr, val = _gen_single_number(rng, digit_len, force_large)
        else:
            expr, val = _gen_base_calculation(rng, digit_len, force_large)
            category = "base_calculation"

        # Apply augmentation with appropriate probability
        if self.enable_augmentation:
            augment_prob = PHASE_AUGMENTATION_PROB.get(self.phase, 0.15)
            
            # EC 카테고리 전용 augmentation 강화 (핵심 개선!)
            if category == "expression_consistency":
                augment_prob *= 1.5  # EC는 1.5배 더 많은 augmentation
                augment_prob = min(augment_prob, 0.80)  # 최대 80%로 제한
            
            if rng.random() < augment_prob:
                augmented = _safe_augment_expression(expr, val, rng)
                if augmented:
                    expr, val = augmented

        return {
            "input_text": expr,
            "target_text": str(val),
            "meta": {
                "category": category,
                "phase": self.phase,
                "digit_len": digit_len,
                "output_6digit": len(str(val)) >= 6,
            },
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """Create DataLoader with validation collate function."""
    is_training = getattr(dataset, "mode", mode) == "train"
    collate_fn = (
        partial(collate_fn_with_validation, is_training=True) if is_training else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
