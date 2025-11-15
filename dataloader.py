"""Arithmetic data generator with category-aware curriculum and comprehensive augmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import random
import re
from functools import partial

from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation

# ---------------------------------------------------------------------------
# Distribution targets (operator-count focused for 1-4 operators)
# ---------------------------------------------------------------------------

# Developer log: Operator count distribution for balanced multi-operator learning
# Fine-tuned based on empirical testing
OPERATOR_COUNT_DISTRIBUTION = {
    1: 0.22,  # 22% - 1개 연산자 (예: a+b, a*b)
    2: 0.35,  # 35% - 2개 연산자 (예: a+b*c, (a+b)+c)
    3: 0.25,  # 25% - 3개 연산자 (예: (a+b)*c-d)
    4: 0.18,  # 18% - 4개 연산자 (예: ((a+b)*c)//d+e)
}

TRAINING_DISTRIBUTION = {
    "base_calculation": 0.10,        # Calculation Accuracy (기본 계산) - EC 강화를 위해 감소
    "precedence": 0.15,              # Calculation Accuracy (연산 우선순위/괄호) - EC 강화를 위해 감소
    "law_preservation": 0.15,        # Law Preservation (교환/결합 법칙) - EC 강화를 위해 감소
    "expression_consistency": 0.35,  # Expression Consistency (표현 일관성) - EC 강화: 0.20 → 0.35 (75% 증가)
    "relational": 0.10,              # Relational Consistency (관계성/항등원) - 유지
    "long_expression": 0.08,         # 긴 수식/연속 연산 집중 - EC 강화를 위해 감소
    "complex_nested": 0.07,          # 복잡 중첩/괄호 패턴 - EC 강화를 위해 감소
}

PHASE_DIGIT_DISTRIBUTION = {
    1: {1: 0.4, 2: 0.6},
    2: {2: 0.4, 3: 0.6},
    3: {3: 0.4, 4: 0.6},
    4: {4: 0.4, 5: 0.6},
}

OUTPUT_6DIGIT_RATIO = {
    "base_calculation": 0.05,        # 기본 계산에서 큰 출력
    "precedence": 0.18,              # 연산 우선순위에서 큰 출력
    "law_preservation": 0.20,        # 법칙 보존에서 큰 출력
    "expression_consistency": 0.30,  # 표현 일관성에서 큰 출력 - EC 강화: 0.25 → 0.30
    "relational": 0.30,              # 관계성에서 큰 출력
}

PHASE_AUGMENTATION_PROB = {
    1: 0.15,  # Phase 1: 15% 증강 (기초 단계)
    2: 0.25,  # Phase 2: 25% 증강 (중급)
    3: 0.30,  # Phase 3: 30% 증강 (고급)
    4: 0.20,  # Phase 4: 20% 증강 (큰 숫자는 증강 줄임)
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


def _count_operators(expr: str) -> int:
    """
    Count number of operators in expression.
    Developer log: Enhanced operator counting for multi-operator expressions.
    """
    tokens = re.findall(r'\d+|//|[+\-*/()]', expr)
    return sum(1 for t in tokens if t in ['+', '-', '*', '//'])


def _has_balanced_parentheses(expr: str) -> bool:
    """
    Check if parentheses are balanced in expression.
    Developer log: Validation utility for augmented expressions.
    """
    cnt = 0
    for ch in expr:
        if ch == '(':
            cnt += 1
        elif ch == ')':
            cnt -= 1
            if cnt < 0:
                return False
    return cnt == 0


def _parse_expression(expr: str) -> List[str]:
    """
    Parse expression into tokens (numbers and operators).
    Developer log: Token parsing for augmentation functions.
    """
    tokens = re.findall(r'\d+|//|[+\-*/()]', expr)
    return tokens


def _sample_operator_count(rng: random.Random) -> int:
    """
    Sample target operator count based on distribution.
    Developer log: Ensures balanced 1-4 operator distribution.
    """
    r = rng.random()
    cumsum = 0.0
    for count, prob in OPERATOR_COUNT_DISTRIBUTION.items():
        cumsum += prob
        if r <= cumsum:
            return count
    return 2  # Default


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
# Advanced augmentation utilities (team8 코드 통합)
# ---------------------------------------------------------------------------


def _find_subexpressions(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Find possible subexpression ranges in token list.
    Developer log: Identifies valid subexpressions for parentheses augmentation.
    """
    subexprs = []
    n = len(tokens)
    
    # 최소 3개 토큰 (숫자 연산자 숫자)
    for i in range(n - 2):
        for j in range(i + 2, n, 2):  # 2씩 증가 (숫자 위치)
            if j < n and tokens[i].isdigit() and tokens[j].isdigit():
                if _is_valid_subexpression(tokens[i:j+1]):
                    subexprs.append((i, j))
    
    return subexprs


def _is_valid_subexpression(tokens: List[str]) -> bool:
    """Check if token sequence is a valid subexpression."""
    if len(tokens) < 3:
        return False
    
    # 숫자로 시작하고 끝나야 함
    if not (tokens[0].isdigit() and tokens[-1].isdigit()):
        return False
    
    # 패턴: 숫자 연산자 숫자 [연산자 숫자]...
    for i in range(len(tokens)):
        if i % 2 == 0:  # 짝수 인덱스는 숫자
            if not tokens[i].isdigit():
                return False
        else:  # 홀수 인덱스는 연산자
            if tokens[i] not in ['+', '-', '*', '//']:
                return False
    
    return True


def augment_with_parentheses(
    expression: str,
    max_augmentations: int = 3
) -> List[str]:
    """
    Generate variations with SAFE parentheses addition.
    Developer log: Only add parentheses when it doesn't change the value.
    Returns list of (expr, value) tuples if value changes, otherwise just expressions.
    """
    if '(' in expression or ')' in expression:
        return []
    
    tokens = _parse_expression(expression)
    
    # Check if only single operator type (safe to add parentheses)
    operators = [t for t in tokens if t in ['+', '-', '*', '//']]
    
    # Only allow parentheses for associative operations (+, *)
    # Subtraction and division are NOT associative
    if operators:
        first_op = operators[0]
        # Safe if all operators are same and associative
        if first_op in ['+', '*'] and all(op == first_op for op in operators):
            # Safe to add parentheses anywhere
            subexprs = _find_subexpressions(tokens)
            
            augmented = []
            for start, end in subexprs[:max_augmentations]:
                new_tokens = tokens[:start] + ['('] + tokens[start:end+1] + [')'] + tokens[end+1:]
                new_expr = ''.join(new_tokens)
                if new_expr != expression:
                    augmented.append(new_expr)
            
            return augmented
    
    # Unsafe to add parentheses (mixed operators or non-associative)
    return []


def augment_with_commutative_advanced(expression: str) -> List[str]:
    """
    Advanced commutative law augmentation.
    Developer log: Swaps operands for + or * only expressions.
    """
    tokens = _parse_expression(expression)
    
    if '(' in expression or ')' in expression or len(tokens) < 3:
        return []
    
    # Check if only + or * operators
    operators = [t for t in tokens if t in ['+', '-', '*', '//']]
    if not all(op in ['+'] for op in operators) and not all(op in ['*'] for op in operators):
        return []
    
    augmented = []
    operand_positions = [i for i in range(0, len(tokens), 2) if tokens[i].isdigit()]
    
    # Swap adjacent operands
    for i in range(len(operand_positions) - 1):
        pos1 = operand_positions[i]
        pos2 = operand_positions[i + 1]
        new_tokens = tokens.copy()
        new_tokens[pos1], new_tokens[pos2] = new_tokens[pos2], new_tokens[pos1]
        new_expr = ''.join(new_tokens)
        if new_expr != expression:
            augmented.append(new_expr)
    
    # Reverse all operands
    if len(operand_positions) > 2:
        reversed_tokens = tokens.copy()
        for i in range(len(operand_positions)):
            orig_pos = operand_positions[i]
            new_pos = operand_positions[-(i+1)]
            reversed_tokens[orig_pos] = tokens[new_pos]
        reversed_expr = ''.join(reversed_tokens)
        if reversed_expr != expression and reversed_expr not in augmented:
            augmented.append(reversed_expr)
    
    return augmented


def augment_with_mathematical_laws(
    expression: str,
    max_augmentations: int = 5,
    max_depth: Optional[int] = None,
    allow_zero_change: bool = False
) -> List[str]:
    """
    Apply comprehensive mathematical law augmentations.
    Developer log: Combines commutative, associative, distributive, parentheses, identity.
    """
    all_augmented = []
    
    # 교환법칙 (advanced version)
    commutative_results = augment_with_commutative_advanced(expression)
    all_augmented.extend(commutative_results)
    
    # 괄호 증강
    parentheses_results = augment_with_parentheses(expression, max_augmentations=3)
    all_augmented.extend(parentheses_results)
    
    # 항등원 증강
    if max_depth is None or _count_operators(expression) < max_depth:
        identity_ops = [
            f"{expression}+0",
            f"0+{expression}",
            f"{expression}-0",
            f"{expression}*1",
            f"1*{expression}",
        ]
        all_augmented.extend(identity_ops[:min(3, max_augmentations)])
    
    # 중복 제거 및 최대 개수 제한
    unique_results = []
    seen = set()
    for expr in all_augmented:
        if expr not in seen:
            seen.add(expr)
            unique_results.append(expr)
        if len(unique_results) >= max_augmentations:
            break
    
    return unique_results


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
    
    # 12% 확률로 긴 수식 생성 (숫자 5개 이상) - 다양성 확대
    if rng.random() < 0.12:
        return _gen_long_expression(rng, digit_len)

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


def _gen_long_expression(
    rng: random.Random,
    digit_len: int,
) -> Tuple[str, int]:
    """
    Generate long expressions with 5-8 operators for improved generalization.
    Developer log: Enhanced to support 5-8 operators with proper precedence handling.
    """
    # 연산자 개수: 5-8개 (기존 5-7 숫자 → 5-8 연산자)
    op_count = rng.randint(5, 8)
    num_count = op_count + 1  # 연산자 개수 + 1 = 숫자 개수
    
    numbers = [_rand_int(rng, (1, min(3, digit_len))) for _ in range(num_count)]
    
    # 연산자 선택 (혼합, 우선순위 고려)
    ops = []
    for _ in range(op_count):
        if rng.random() < 0.5:
            ops.append(rng.choice(["+", "-"]))
        else:
            ops.append(rng.choice(["*", "//"]))
    
    # 수식 구성
    expr_parts = [str(numbers[0])]
    val = numbers[0]
    
    for i, op in enumerate(ops):
        num = numbers[i + 1]
        if op == "//":
            if num == 0:
                num = rng.randint(1, 9)
            expr_parts.append(f"//{num}")
            val = val // num
        elif op == "*":
            expr_parts.append(f"*{num}")
            val = val * num
        elif op == "+":
            expr_parts.append(f"+{num}")
            val = val + num
        else:  # "-"
            if val >= num:
                expr_parts.append(f"-{num}")
                val = val - num
            else:
                expr_parts.append(f"+{num}")
                val = val + num
    
    return "".join(expr_parts), val


def _gen_law_preservation(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate expressions that test law preservation (commutative, associative).
    Developer log: 교환법칙, 결합법칙 테스트를 위한 데이터 생성.
    """
    if force_large:
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        c = _rand_int(rng, (2, 3))
        pattern = rng.choice(["commutative", "associative"])
        if pattern == "commutative":
            # 교환법칙: a+b or a*b (순서만 다름)
            if rng.random() < 0.5:
                return f"{a}+{b}", a + b  # 다른 곳에서 b+a 생성될 것
            return f"{a}*{b}", a * b
        else:
            # 결합법칙: (a+b)+c or a+(b+c)
            if rng.random() < 0.5:
                return f"({a}+{b})+{c}", (a + b) + c
            return f"{a}*({b}+{c})", a * (b + c)
    
    pattern_choice = rng.random()
    
    if pattern_choice < 0.4:
        # 교환법칙 테스트: 덧셈
        a = _rand_int(rng, (1, min(3, digit_len)))
        b = _rand_int(rng, (1, min(3, digit_len)))
        if rng.random() < 0.5:
            return f"{a}+{b}", a + b
        return f"{b}+{a}", b + a
    
    elif pattern_choice < 0.7:
        # 교환법칙 테스트: 곱셈
        a = _rand_int(rng, (1, min(2, digit_len)))
        b = _rand_int(rng, (1, min(2, digit_len)))
        if rng.random() < 0.5:
            return f"{a}*{b}", a * b
        return f"{b}*{a}", b * a
    
    else:
        # 결합법칙 테스트
        a = _rand_int(rng, (1, min(2, digit_len)))
        b = _rand_int(rng, (1, min(2, digit_len)))
        c = _rand_int(rng, (1, min(2, digit_len)))
        
        if rng.random() < 0.5:
            # 덧셈 결합법칙
            if rng.random() < 0.5:
                return f"({a}+{b})+{c}", (a + b) + c
            return f"{a}+({b}+{c})", a + (b + c)
        else:
            # 곱셈 결합법칙
            if rng.random() < 0.5:
                return f"({a}*{b})*{c}", (a * b) * c
            return f"{a}*({b}*{c})", a * (b * c)


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
    Developer log: EC 강화 - 단순 이항 연산 비중 증가, 순서 랜덤화, 결합법칙 괄호 위치 랜덤화
    """
    if force_large:
        a = _rand_int(rng, (4, 5))
        b = _rand_int(rng, (2, 3))
        # EC 강화: 순서 랜덤화 (50/50)
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                return f"{a}*{b}", a * b
            return f"{b}*{a}", b * a
        else:
            if rng.random() < 0.5:
                return f"{a}+{b}", a + b
            return f"{b}+{a}", b + a

    # EC 강화: 단순 이항 연산 비중 60% → 75%
    if rng.random() < 0.75:
        op = rng.choice(["+", "*"])
        a = _rand_int(rng, (1, digit_len))
        b = _rand_int(rng, (1, digit_len))
        # EC 강화: 순서 랜덤화 (50/50) - A+B vs B+A 균등 생성
        if rng.random() < 0.5:
            expr = f"{a}{op}{b}"
        else:
            expr = f"{b}{op}{a}"
        val = a + b if op == "+" else a * b
        return expr, val

    # 결합법칙 패턴 (25%)
    op = rng.choice(["+", "*"])
    a = _rand_int(rng, (1, digit_len))
    b = _rand_int(rng, (1, digit_len))
    c = _rand_int(rng, (1, digit_len))
    # EC 강화: 결합법칙 괄호 위치 랜덤화 (50/50) - (A+B)+C vs A+(B+C) 균등 생성
    if rng.random() < 0.5:
        # (A op B) op C 형태
        if op == "+":
            return f"({a}+{b})+{c}", (a + b) + c
        return f"({a}*{b})*{c}", (a * b) * c
    else:
        # A op (B op C) 형태
        if op == "+":
            return f"{a}+({b}+{c})", a + (b + c)
        return f"{a}*({b}*{c})", a * (b * c)


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


def _gen_complex_nested(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate complex nested expressions by composing existing generators.
    Developer log: Mixes precedence, law, consistency segments with additional
    outer operations to maximize structural diversity.
    """
    segment_generators = [
        _gen_precedence,
        _gen_law_preservation,
        _gen_expression_consistency_base,
        _gen_relational,
        _gen_base_calculation,
    ]
    seg_count = 4 if force_large else rng.randint(3, 4)
    segments: List[Tuple[str, int]] = []
    for _ in range(seg_count):
        gen = rng.choice(segment_generators)
        seg_expr, seg_val = gen(
            rng,
            min(5, digit_len + (1 if force_large else 0)),
            force_large,
        )
        segments.append((seg_expr, seg_val))
    
    expr, value = segments[0]
    for seg_expr, seg_val in segments[1:]:
        op = rng.choice(["+", "-", "*"])
        if op == "-":
            if value < seg_val:
                expr, seg_expr = seg_expr, expr
                value, seg_val = seg_val, value
            value -= seg_val
        elif op == "+":
            value += seg_val
        else:
            value *= seg_val
        expr = f"({expr}){op}({seg_expr})"
    
    if rng.random() < 0.5:
        booster = _rand_int(rng, (3, 5)) if force_large else _rand_int(rng, (1, digit_len))
        if rng.random() < 0.5:
            expr = f"{booster}*({expr})"
            value *= booster
        else:
            expr = f"({expr})+{booster}"
            value += booster
    
    return expr, value


def _gen_single_number(
    rng: random.Random,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """Generate single number (for curriculum completeness)."""
    val = _rand_int(rng, (5, 5)) if force_large else _rand_int(rng, (1, digit_len))
    text = str(val)
    return text, val


def _gen_expr_with_target_op_count(
    rng: random.Random,
    target_op_count: int,
    digit_len: int,
    force_large: bool = False,
) -> Tuple[str, int]:
    """
    Generate expression with specific operator count.
    Developer log: Core function for balanced operator distribution (1-4 operators).
    
    Args:
        target_op_count: Desired number of operators (1-4)
        digit_len: Maximum digit length
        force_large: Force large output (6+ digits)
    
    Returns:
        (expression, value) tuple
    """
    # Map operator count to appropriate depth
    depth_map = {1: 1, 2: 2, 3: 3, 4: 4}
    target_depth = depth_map.get(target_op_count, 2)
    
    # Try up to 15 times to hit target operator count
    for attempt in range(15):
        # Select category based on operator count
        if target_op_count == 1:
            # Simple binary operations
            if rng.random() < 0.5:
                expr, val = _gen_base_calculation(rng, digit_len, force_large)
            else:
                expr, val = _gen_law_preservation(rng, digit_len, force_large)
        
        elif target_op_count == 2:
            # 2 operators: precedence or simple chains
            category_choice = rng.random()
            if category_choice < 0.4:
                expr, val = _gen_precedence(rng, digit_len, force_large)
            elif category_choice < 0.7:
                expr, val = _gen_base_calculation(rng, digit_len, force_large)
            else:
                expr, val = _gen_expression_consistency_base(rng, digit_len, force_large)
        
        elif target_op_count == 3:
            # 3 operators: ALWAYS generate manually for accurate count
            a = _rand_int(rng, (1, digit_len))
            b = _rand_int(rng, (1, digit_len))
            c = _rand_int(rng, (1, digit_len))
            d = _rand_int(rng, (1, digit_len))
            
            # Various patterns for 3 operators
            pattern = rng.choice(['chain', 'nested', 'mixed'])
            
            if pattern == 'chain':
                # Simple chain: a op1 b op2 c op3 d
                ops = [rng.choice(['+', '-', '*']) for _ in range(3)]
                val = a
                expr = str(a)
                
                for i, (op, num) in enumerate(zip(ops, [b, c, d])):
                    if op == '*':
                        val *= num
                        expr += f"*{num}"
                    elif op == '+':
                        val += num
                        expr += f"+{num}"
                    else:  # '-'
                        if val >= num:
                            val -= num
                            expr += f"-{num}"
                        else:
                            val += num
                            expr += f"+{num}"
            
            elif pattern == 'nested':
                # Pattern: (a op1 b) op2 c op3 d
                op1, op2, op3 = rng.choice(['+', '*']), rng.choice(['+', '-']), rng.choice(['+', '-'])
                
                if op1 == '+':
                    val1 = a + b
                else:
                    val1 = a * b
                
                val = val1
                expr = f"({a}{op1}{b})"
                
                for op, num in [(op2, c), (op3, d)]:
                    if op == '+':
                        val += num
                        expr += f"+{num}"
                    else:
                        if val >= num:
                            val -= num
                            expr += f"-{num}"
                        else:
                            val += num
                            expr += f"+{num}"
            
            else:  # mixed
                # Pattern: a op1 (b op2 c) op3 d
                op1, op2, op3 = rng.choice(['+', '*']), rng.choice(['+', '*']), rng.choice(['+', '-'])
                
                if op2 == '+':
                    val2 = b + c
                else:
                    val2 = b * c
                
                if op1 == '+':
                    val = a + val2
                else:
                    val = a * val2
                
                if op3 == '+':
                    val += d
                    op3_used = '+'
                else:
                    if val >= d:
                        val -= d
                        op3_used = '-'
                    else:
                        val += d
                        op3_used = '+'
                
                expr = f"{a}{op1}({b}{op2}{c}){op3_used}{d}"
        
        elif target_op_count >= 4:
            # 4 operators: ALWAYS generate manually for accurate count
            nums = [_rand_int(rng, (1, min(3, digit_len))) for _ in range(5)]
            
            # Various patterns for 4 operators
            pattern = rng.choice(['chain', 'nested', 'mixed'])
            
            if pattern == 'chain':
                # Simple chain: a op1 b op2 c op3 d op4 e
                ops = [rng.choice(['+', '-', '*']) for _ in range(4)]
                val = nums[0]
                expr = str(nums[0])
                
                for op, num in zip(ops, nums[1:]):
                    if op == '*':
                        val *= num
                        expr += f"*{num}"
                    elif op == '+':
                        val += num
                        expr += f"+{num}"
                    else:  # '-'
                        if val >= num:
                            val -= num
                            expr += f"-{num}"
                        else:
                            val += num
                            expr += f"+{num}"
            
            elif pattern == 'nested':
                # Pattern: ((a op1 b) op2 c) op3 d op4 e
                ops = [rng.choice(['+', '*']) for _ in range(2)] + [rng.choice(['+', '-']) for _ in range(2)]
                
                # First pair
                if ops[0] == '+':
                    val = nums[0] + nums[1]
                else:
                    val = nums[0] * nums[1]
                expr = f"({nums[0]}{ops[0]}{nums[1]})"
                
                # Second operation
                if ops[1] == '+':
                    val += nums[2]
                    expr = f"({expr}+{nums[2]})"
                else:
                    val *= nums[2]
                    expr = f"({expr}*{nums[2]})"
                
                # Remaining operations
                for op, num in zip(ops[2:], nums[3:]):
                    if op == '+':
                        val += num
                        expr += f"+{num}"
                    else:
                        if val >= num:
                            val -= num
                            expr += f"-{num}"
                        else:
                            val += num
                            expr += f"+{num}"
            
            else:  # mixed
                # Pattern: a op1 (b op2 c op3 d) op4 e
                ops = [rng.choice(['+', '*']), rng.choice(['+', '*']), rng.choice(['+', '-']), rng.choice(['+', '-'])]
                
                # Inner expression: b op2 c op3 d
                if ops[1] == '+':
                    val_inner = nums[1] + nums[2]
                else:
                    val_inner = nums[1] * nums[2]
                
                if ops[2] == '+':
                    val_inner += nums[3]
                    expr_inner = f"{nums[1]}{ops[1]}{nums[2]}+{nums[3]}"
                else:
                    if val_inner >= nums[3]:
                        val_inner -= nums[3]
                        expr_inner = f"{nums[1]}{ops[1]}{nums[2]}-{nums[3]}"
                    else:
                        val_inner += nums[3]
                        expr_inner = f"{nums[1]}{ops[1]}{nums[2]}+{nums[3]}"
                
                # First operation with inner
                if ops[0] == '+':
                    val = nums[0] + val_inner
                else:
                    val = nums[0] * val_inner
                expr = f"{nums[0]}{ops[0]}({expr_inner})"
                
                # Last operation
                if ops[3] == '+':
                    val += nums[4]
                    expr += f"+{nums[4]}"
                else:
                    if val >= nums[4]:
                        val -= nums[4]
                        expr += f"-{nums[4]}"
                    else:
                        val += nums[4]
                        expr += f"+{nums[4]}"
        
        # Verify operator count
        actual_count = _count_operators(expr)
        
        # Accept if within ±1 of target
        if abs(actual_count - target_op_count) <= 1:
            return expr, val
    
    # Fallback: use base calculation
    return _gen_base_calculation(rng, digit_len, force_large)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ArithmeticDataset(Dataset):
    """
    Arithmetic dataset with category-aware curriculum and comprehensive augmentation.
    Developer log: Supports phase, phase_mix, max_depth parameters for backward compatibility.
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
        phase_mix: Optional[Tuple[int, ...]] = None,
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
        if phase_mix is not None and len(phase_mix) > 0:
            normalized = tuple(
                sorted(
                    {
                        max(1, min(4, int(p)))
                        for p in phase_mix
                    }
                )
            )
            self.phase_pool = normalized or (phase,)
            self.phase = self.phase_pool[0]
        else:
            self.phase = phase
            self.phase_pool = (self.phase,)

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

    def _sample_phase(self, rng: random.Random) -> int:
        if len(self.phase_pool) == 1:
            return self.phase_pool[0]
        return rng.choice(self.phase_pool)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        
        # Developer log: Operator-count focused generation for balanced distribution
        # 80% operator-count controlled, 20% special categories (long_expr, etc.)
        use_op_count_control = rng.random() < 0.80
        
        sample_phase = self._sample_phase(rng)
        digit_len = _sample_digit_length(rng, sample_phase)
        
        if use_op_count_control:
            # Operator count based generation (1-4 operators)
            target_op_count = _sample_operator_count(rng)
            category = f"op{target_op_count}_expr"
            force_large = rng.random() < 0.10  # 10% large output
            
            expr, val = _gen_expr_with_target_op_count(
                rng, target_op_count, digit_len, force_large
            )
        else:
            # Special categories (long, complex nested, relational)
            category = self._sample_category(rng)
            force_large = _should_force_large_output(rng, category)
            
            if category == "long_expression":
                expr, val = _gen_long_expression(rng, digit_len + 1 if force_large else digit_len)
            elif category == "complex_nested":
                expr, val = _gen_complex_nested(rng, digit_len, force_large)
            elif category == "relational":
                expr, val = _gen_relational(rng, digit_len, force_large)
            else:
                # Fallback to operator-count based
                target_op_count = _sample_operator_count(rng)
                expr, val = _gen_expr_with_target_op_count(
                    rng, target_op_count, digit_len, force_large
                )
                category = f"op{target_op_count}_expr"

        # Apply augmentation with appropriate probability
        # Developer log: EC 강화 - expression_consistency 카테고리에 augmentation 확률 1.5배 boost
        if self.enable_augmentation:
            augment_prob = PHASE_AUGMENTATION_PROB.get(sample_phase, 0.15)
            # EC 카테고리에 augmentation 확률 boost (EC 강화 전략)
            if category == "expression_consistency":
                augment_prob *= 1.5  # EC는 1.5배 boost
                augment_prob = min(augment_prob, 0.80)  # 최대 80%로 제한
            if rng.random() < augment_prob:
                augmented = _safe_augment_expression(expr, val, rng)
                if augmented:
                    expr, val = augmented
        
        # CRITICAL: Recompute target to ensure correctness (fixes operator precedence bugs)
        # Developer log: Original generators use sequential calculation, but expressions follow precedence
        try:
            actual_value = eval(expr.replace('//', '//'))
            if actual_value < 0:
                # Regenerate if negative result
                # Fallback to simple expression
                a = _rand_int(rng, (1, digit_len))
                b = _rand_int(rng, (1, digit_len))
                expr = f"{a}+{b}"
                actual_value = a + b
            val = actual_value
        except:
            # If eval fails, fallback to simple expression
            a = _rand_int(rng, (1, digit_len))
            b = _rand_int(rng, (1, digit_len))
            expr = f"{a}+{b}"
            val = a + b

        return {
            "input_text": expr,
            "target_text": str(val),
            "meta": {
                "category": category,
                "phase": sample_phase,
                "digit_len": digit_len,
                "output_6digit": len(str(val)) >= 6,
            },
        }


def create_augmented_dataset_from_original(original_dataset: Dataset) -> List[Dict[str, Any]]:
    """
    Create augmented dataset with group_id for EC learning.
    Developer log: Generates expression pairs from original dataset for Expression Consistency.
    Returns original + augmented samples with group_id field.
    Validates mathematical correctness by recomputing all values.
    """
    augmented_data = []
    
    print(f"Creating augmented dataset from {len(original_dataset)} samples...")
    
    for idx in range(len(original_dataset)):
        original_item = original_dataset[idx]
        expr = original_item["input_text"]
        target = original_item["target_text"]
        
        # Validate original - recompute to ensure correctness
        target_str = str(target)
        if not (len(target_str) > 0 and target_str.isdigit()):
            continue
        if not _has_balanced_parentheses(expr):
            continue
        
        # CRITICAL: Recompute original expression to verify correctness
        try:
            actual_value = eval(expr.replace('//', '//'))
            if str(actual_value) != target_str or actual_value < 0:
                # Skip if mismatch or negative
                continue
        except:
            continue
        
        # Group ID for EC learning
        group_id = f"ec_group_{idx}"
        
        # Add original with group_id
        original_with_group = {
            "input_text": expr,
            "target_text": target_str,
            "group_id": group_id,
            "meta": original_item.get("meta", {}).copy(),
        }
        original_with_group["meta"]["augmented"] = False
        # Developer log: EC 학습을 위해 meta에도 group_id 저장
        original_with_group["meta"]["group_id"] = group_id
        augmented_data.append(original_with_group)
        
        # Generate augmentations
        # Developer log: 괄호/법칙/관계성 카테고리는 더 많은 동치 수식 쌍을 생성
        meta = original_item.get("meta", {})
        cat = meta.get("category", "")
        if cat in {
            "precedence",
            "law_preservation",
            "expression_consistency",
            "relational",
            "complex_nested",
        }:
            max_augs = 7
        else:
            max_augs = 5
        augmented_exprs = augment_with_mathematical_laws(
            expr,
            max_augmentations=max_augs,
            max_depth=None,
            allow_zero_change=False,
        )
        
        for aug_expr in augmented_exprs:
            if isinstance(aug_expr, tuple):
                expr_str, new_target = aug_expr
            else:
                expr_str, new_target = aug_expr, target_str
            
            new_target_str = str(new_target)
            
            # Validate augmented
            if not (len(new_target_str) > 0 and new_target_str.isdigit()):
                continue
            if not _has_balanced_parentheses(expr_str):
                continue
            
            # CRITICAL: Recompute augmented expression to verify correctness
            try:
                actual_aug_value = eval(expr_str.replace('//', '//'))
                if str(actual_aug_value) != new_target_str or actual_aug_value < 0:
                    # Skip if mismatch or negative
                    continue
            except:
                continue
            
            aug_item = {
                "input_text": expr_str,
                "target_text": new_target_str,
                "group_id": group_id,  # Same group_id for EC
                "meta": {**original_with_group["meta"], "augmented": True},
            }
            # Developer log: EC consistency를 위해 meta에도 group_id 명시적으로 보존
            aug_item["meta"]["group_id"] = group_id
            augmented_data.append(aug_item)
    
    # Remove duplicates
    seen = set()
    unique_data = []
    for item in augmented_data:
        key = (item["input_text"], item["target_text"])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    print(f"Augmentation complete: {len(original_dataset)} → {len(unique_data)} samples")
    return unique_data


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """Create DataLoader with validation collate function."""
    is_training = getattr(dataset, "mode", mode) == "train"
    # Developer log: validation 모드에서도 collate_fn_with_validation 사용하여 meta 정보 보존
    collate_fn = partial(collate_fn_with_validation, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
