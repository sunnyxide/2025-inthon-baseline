"""Enhanced data generation with category-based distribution and augmentation"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import random
import re
from functools import partial
from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation


# Category distribution (v0 baseline)
TRAINING_DISTRIBUTION = {
    "base_calculation": 0.40,
    "precedence": 0.20,
    "expression_consistency": 0.25,
    "relational": 0.10,
    "single_number": 0.05,
}

# Phase-based digit length distribution (within each phase)
PHASE_DIGIT_DISTRIBUTION = {
    1: {1: 0.4, 2: 0.6},  # Phase 1: 1-2 digits
    2: {2: 0.4, 3: 0.6},  # Phase 2: 2-3 digits
    3: {3: 0.4, 4: 0.6},  # Phase 3: 3-4 digits
    4: {4: 0.4, 5: 0.6},  # Phase 4: 4-5 digits
}

# Output 6+ digit ratio per category
OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.05,
    "precedence": 0.10,
    "expression_consistency": 0.20,
    "relational": 0.30,
    "single_number": 0.0,
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
    return "base_calculation"  # fallback


def _sample_digit_length(rng: random.Random, phase: int) -> int:
    """Sample digit length based on phase distribution"""
    dist = PHASE_DIGIT_DISTRIBUTION.get(phase, {4: 0.5, 5: 0.5})
    r = rng.random()
    cumsum = 0.0
    for digits, prob in dist.items():
        cumsum += prob
        if r <= cumsum:
            return digits
    return 4  # fallback


def _should_generate_6digit_output(rng: random.Random, category: str) -> bool:
    """Check if we should generate 6+ digit output for this category"""
    ratio = OUTPUT_6DIGIT_RATIO.get(category, 0.0)
    return rng.random() < ratio


def _gen_base_calculation(
    rng: random.Random, digit_len: int, force_6digit_output: bool = False
) -> Tuple[str, int]:
    """Generate base calculation expressions (no parentheses, no augmentation)"""
    # Operator distribution within base_calculation
    op_weights = {"+": 0.40, "-": 0.25, "*": 0.25, "//": 0.10}
    
    if force_6digit_output:
        # Force large output: use multiplication or large addition
        if rng.random() < 0.7:
            # Large multiplication: 4-5 digit * 2-3 digit
            a = _rand_int(rng, (4, 5))
            b = _rand_int(rng, (2, 3))
            expr = f"{a}*{b}"
            val = a * b
        else:
            # Large addition: 4-5 digit + 4-5 digit
            a = _rand_int(rng, (4, 5))
            b = _rand_int(rng, (4, 5))
            expr = f"{a}+{b}"
            val = a + b
    else:
        # Normal generation: 2-4 terms
        num_terms = rng.randint(2, 4)
        terms = []
        values = []
        
        for i in range(num_terms):
            v = _rand_int(rng, (1, digit_len))
            terms.append(str(v))
            values.append(v)
        
        # Select operators
        ops = []
        for i in range(num_terms - 1):
            op = rng.choices(
                list(op_weights.keys()),
                weights=list(op_weights.values()),
                k=1
            )[0]
            ops.append(op)
        
        # Build expression respecting operator precedence
        # Group * and // first, then + and -
        expr = terms[0]
        val = values[0]
        
        for i, op in enumerate(ops):
            next_val = values[i + 1]
            if op == "+":
                expr = f"{expr}+{terms[i+1]}"
                val = val + next_val
            elif op == "-":
                if val < next_val:
                    # Swap to avoid negative
                    expr = f"{terms[i+1]}-{expr}"
                    val = next_val - val
                else:
                    expr = f"{expr}-{terms[i+1]}"
                    val = val - next_val
            elif op == "*":
                expr = f"{expr}*{terms[i+1]}"
                val = val * next_val
            elif op == "//":
                if next_val == 0:
                    next_val = rng.randint(1, 9)
                    terms[i+1] = str(next_val)
                expr = f"{expr}//{terms[i+1]}"
                val = val // next_val
    
    return expr, val


def _gen_precedence(
    rng: random.Random, digit_len: int, force_6digit_output: bool = False
) -> Tuple[str, int]:
    """Generate expressions with parentheses to test precedence"""
    if force_6digit_output:
        # Large output with parentheses
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        c = _rand_int(rng, (2, 3))
        if rng.random() < 0.5:
            expr = f"({a}+{b})*{c}"
            val = (a + b) * c
        else:
            expr = f"{a}*({b}+{c})"
            val = a * (b + c)
    else:
        # Normal precedence patterns
        patterns = [
            # Pattern 1: (a+b)*c vs a+b*c
            lambda: (
                f"({_rand_int(rng, (1, digit_len))}+{_rand_int(rng, (1, digit_len))})*{_rand_int(rng, (1, digit_len))}",
                lambda a, b, c: (a + b) * c
            ),
            # Pattern 2: a*(b+c) vs a*b+c
            lambda: (
                f"{_rand_int(rng, (1, digit_len))}*({_rand_int(rng, (1, digit_len))}+{_rand_int(rng, (1, digit_len))})",
                lambda a, b, c: a * (b + c)
            ),
            # Pattern 3: Nested parentheses
            lambda: (
                f"(({_rand_int(rng, (1, digit_len))}+{_rand_int(rng, (1, digit_len))})*{_rand_int(rng, (1, digit_len))})+{_rand_int(rng, (1, digit_len))}",
                lambda a, b, c, d: ((a + b) * c) + d
            ),
        ]
        
        pattern = rng.choice(patterns)
        expr_template, val_fn = pattern()
        
        # Extract numbers and compute value
        numbers = [int(x) for x in re.findall(r'\d+', expr_template)]
        val = val_fn(*numbers)
        expr = expr_template
    
    return expr, val


def _gen_expression_consistency_base(
    rng: random.Random, digit_len: int, force_6digit_output: bool = False
) -> Tuple[str, int]:
    """Generate base expression for consistency (commutative/associative laws)"""
    if force_6digit_output:
        # Large commutative pairs
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        if rng.random() < 0.5:
            expr = f"{a}*{b}"
            val = a * b
        else:
            expr = f"{a}+{b}"
            val = a + b
    else:
        # Simple 2-term expressions (will be augmented)
        op = rng.choice(["+", "*"])
        a = _rand_int(rng, (1, digit_len))
        b = _rand_int(rng, (1, digit_len))
        expr = f"{a}{op}{b}"
        if op == "+":
            val = a + b
        elif op == "*":
            val = a * b
        else:
            val = 0  # fallback
    
    return expr, val


def _safe_augment_expression(expr: str, val: int, rng: random.Random) -> Optional[Tuple[str, int]]:
    """Safely augment expression using commutative/associative laws (only for expression_consistency)"""
    # Only augment simple 2-term expressions without parentheses
    if "(" in expr or ")" in expr:
        return None
    
    # Tokenize: numbers and operators
    tokens = re.findall(r'\d+|[\+\-\*//]', expr)
    if len(tokens) != 3:  # num op num
        return None
    
    num1_str, op, num2_str = tokens
    
    # Only commutative operations
    if op in ["+", "*"]:
        try:
            num1 = int(num1_str)
            num2 = int(num2_str)
            # Commutative: a+b -> b+a, a*b -> b*a
            new_expr = f"{num2}{op}{num1}"
            if op == "+":
                new_val = num2 + num1
            elif op == "*":
                new_val = num2 * num1
            else:
                return None
            
            if new_val == val:  # Verify equivalence
                return new_expr, new_val
        except (ValueError, TypeError):
            pass
    
    # Associative: (a+b)+c -> a+(b+c) for 3+ terms (future extension)
    # For now, only commutative
    
    return None


def _gen_relational(
    rng: random.Random, digit_len: int, force_6digit_output: bool = False
) -> Tuple[str, int]:
    """Generate relational expressions (A+0, A*1, A+1, etc.)"""
    base_val = _rand_int(rng, (1, digit_len))
    
    if force_6digit_output:
        # Use large base value
        base_val = _rand_int(rng, (4, 5))
    
    patterns = [
        # Identity: A+0, 0+A
        (f"{base_val}+0", base_val),
        (f"0+{base_val}", base_val),
        # Identity: A*1, 1*A
        (f"{base_val}*1", base_val),
        (f"1*{base_val}", base_val),
        # Zero: A*0, 0*A
        (f"{base_val}*0", 0),
        (f"0*{base_val}", 0),
        # Increment: A+1, A+2
        (f"{base_val}+1", base_val + 1),
        (f"{base_val}+2", base_val + 2),
    ]
    
    expr, val = rng.choice(patterns)
    
    # For 6-digit output, ensure result is large
    if force_6digit_output and val < 100000:
        # Use multiplication to force large output
        multiplier = rng.randint(2, 20)
        expr = f"{base_val}*{multiplier}"
        val = base_val * multiplier
    
    return expr, val


def _gen_single_number(
    rng: random.Random, digit_len: int, force_6digit_output: bool = False
) -> Tuple[str, int]:
    """Generate single number (no operation)"""
    if force_6digit_output:
        # 5-digit number (max allowed input)
        val = _rand_int(rng, (5, 5))
    else:
        val = _rand_int(rng, (1, digit_len))
    expr = str(val)
    return expr, val


class ArithmeticDataset(Dataset):
    """Enhanced arithmetic dataset with category-based distribution and augmentation"""
    
    def __init__(
        self,
        num_samples: int,
        phase: int = 4,  # Phase 1-4: (1-2), (2-3), (3-4), (4-5) digits
        seed: int = 42,
        mode: str = "train",
        enable_augmentation: bool = True,  # Enable augmentation for expression_consistency
    ):
        """
        Args:
            num_samples: Total number of samples
            phase: Phase number (1-4) determining digit length distribution
            seed: Random seed
            mode: "train" or "val"
            enable_augmentation: Whether to apply augmentation for expression_consistency
        """
        self.num_samples = num_samples
        self.phase = phase
        self.seed = seed
        self.mode = mode
        self.enable_augmentation = enable_augmentation
        
        # Pre-compute category cumulative distribution for efficient sampling
        self._category_cumsum = []
        cumsum = 0.0
        for cat, prob in TRAINING_DISTRIBUTION.items():
            cumsum += prob
            self._category_cumsum.append((cumsum, cat))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        
        # 1) Sample category
        category = _sample_category(rng)
        
        # 2) Sample digit length based on phase
        digit_len = _sample_digit_length(rng, self.phase)
        
        # 3) Check if we should generate 6+ digit output
        force_6digit = _should_generate_6digit_output(rng, category)
        
        # 4) Generate expression based on category
        if category == "base_calculation":
            expr, val = _gen_base_calculation(rng, digit_len, force_6digit)
        elif category == "precedence":
            expr, val = _gen_precedence(rng, digit_len, force_6digit)
        elif category == "expression_consistency":
            expr, val = _gen_expression_consistency_base(rng, digit_len, force_6digit)
            # Apply augmentation with 50% probability
            if self.enable_augmentation and rng.random() < 0.5:
                augmented = _safe_augment_expression(expr, val, rng)
                if augmented is not None:
                    expr, val = augmented
        elif category == "relational":
            expr, val = _gen_relational(rng, digit_len, force_6digit)
        elif category == "single_number":
            expr, val = _gen_single_number(rng, digit_len, force_6digit)
        else:
            # Fallback to base_calculation
            expr, val = _gen_base_calculation(rng, digit_len, force_6digit)
            category = "base_calculation"
        
        return {
            "input_text": expr,
            "target_text": str(val),
            "meta": {
                "category": category,
                "phase": self.phase,
                "digit_len": digit_len,
                "output_6digit": len(str(val)) >= 6,
            }
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """Create DataLoader with automatic validation for training mode"""
    is_training = (getattr(dataset, "mode", mode) == "train")
    collate_fn = partial(collate_fn_with_validation, is_training=is_training) if is_training else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
