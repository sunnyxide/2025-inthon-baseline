from __future__ import annotations



from typing import List, Any, Tuple, Optional

from config import (
    TrainConfig,
    ModelConfig,
    TokenizerConfig,
    DEPTH_PROFILES,
    apply_depth_profile,
)

import os
import re
import math

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, IterableDataset, Dataset

from tqdm import tqdm

import wandb

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
    create_augmented_dataset_from_original,  # 증강 데이터셋 생성 함수
)

from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (

    TinySeq2Seq,

    TransformerSeq2Seq,

    CharTokenizer,      # 문자 단위 토크나이저

    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수

    INPUT_CHARS,        # 입력 문자 집합

    OUTPUT_CHARS,       # 출력 문자 집합

)

RPN_EXTRA_CHARS = [" ", "+", "-", "*", "D"]  # D는 // 연산자를 나타냄


def _merge_chars(base_chars: List[str]) -> List[str]:
    """
    Merge base output chars with RPN operators.
    Developer log: RPN tokenizer needs space + operators for "12 3 + 4 *" format.
    """
    seen = set()
    merged: List[str] = []
    for ch in base_chars + RPN_EXTRA_CHARS:
        if ch not in seen:
            merged.append(ch)
            seen.add(ch)
    return merged


def build_rpn_tokenizer(tokenizer_config: TokenizerConfig) -> CharTokenizer:
    """
    Build RPN tokenizer with extended vocab (digits + space + operators).
    Developer log: Supports char-level RPN encoding, avoiding multi-char token issues.
    """
    base_chars = (
        tokenizer_config.output_chars
        if tokenizer_config.output_chars is not None
        else OUTPUT_CHARS
    )
    merged_chars = _merge_chars(list(base_chars))
    return CharTokenizer(merged_chars, add_special=tokenizer_config.add_special)


def _encode_rpn_text(tokenizer: CharTokenizer, text: str) -> List[int]:
    """
    Helper for safer RPN encoding with clearer error messages.
    """
    try:
        return tokenizer.encode(text, add_bos_eos=False)
    except ValueError as exc:
        unknown_chars = sorted({ch for ch in set(text) if ch not in tokenizer.stoi})
        raise ValueError(
            f"RPN tokenizer missing chars {unknown_chars} for text '{text}'"
        ) from exc


def _tokenize_infix_for_rpn(expr: str) -> List[str]:
    """
    Infix expression을 RPN stack value 계산용 토큰 리스트로 변환하기 위한 간단 토크나이저.
    Developer log: 숫자/연산자/괄호 단위로 분리 (multi-digit 숫자 유지).
    """
    expr = expr.replace(" ", "")
    return re.findall(r"\d+|//|[+\-*/()]", expr)


def _infix_to_rpn_numbers(expr: str) -> List[str]:
    """
    숫자 토큰 수준의 RPN 시퀀스 생성 (stack value 계산용).
    Developer log: 기존 char-level RPN과 동일한 연산자 순서를 보장.
    """
    tokens = _tokenize_infix_for_rpn(expr)
    output: List[str] = []
    stack: List[str] = []

    def precedence(op: str) -> int:
        if op in ("*", "//"):
            return 2
        if op in ("+", "-"):
            return 1
        return 0

    for tok in tokens:
        if tok.isdigit():
            output.append(tok)
        elif tok in {"+", "-", "*", "//"}:
            while stack and stack[-1] not in "(" and precedence(stack[-1]) >= precedence(tok):
                output.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()

    while stack:
        op = stack.pop()
        if op != "(":
            output.append(op)
    return output


def _compute_rpn_stack_values(expr: str) -> List[float]:
    """
    Infix 식에서 RPN 스택 중간 값(연산자 적용 직후 top 값)의 log-scale 리스트를 계산.
    Developer log: scratchpad supervision용 label 생성 (train 전용).
    음수 결과는 데이터 생성 단계에서 이미 필터링되므로 발생하지 않음 (제3조 ②항).
    """
    rpn_tokens = _infix_to_rpn_numbers(expr)
    stack: List[int] = []
    values: List[float] = []
    for tok in rpn_tokens:
        if tok.isdigit():
            stack.append(int(tok))
        elif tok in {"+", "-", "*", "//"} and len(stack) >= 2:
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                res = a + b
            elif tok == "-":
                res = a - b  # 음수는 데이터 생성에서 이미 방지됨
            elif tok == "*":
                res = a * b
            else:  # "//"
                if b == 0:
                    # 방어적 처리: 0으로 나누기 발생 시 결과를 0으로 클램핑
                    res = 0
                else:
                    res = a // b
            stack.append(res)
            # log-scale 값으로 변환하여 범위 안정화
            values.append(math.log10(res + 1.0))
        # 기타 토큰은 무시
    return values


def compute_ec_consistency_loss(
    logits: torch.Tensor,
    target_output: torch.Tensor,
    meta_list: List[dict] | None,
    pad_id: int,
    lambda_consistency: float,
) -> torch.Tensor | None:
    """
    Expression Consistency 강화를 위한 동치 수식 그룹 consistency loss.
    
    Developer log: 같은 group_id를 가진 동치 수식 쌍/트리플의 출력 분포를 일치시킴.
    Law Preservation / Expression Consistency 지표 직접 개선용.

    Args:
        logits: (B, T, V) 모델 출력 (result_logits)
        target_output: (B, T) target token ids (pad 포함)
        meta_list: batch["meta"] (각 샘플별 dict, group_id 포함)
        pad_id: 출력 토크나이저 pad_id
        lambda_consistency: 손실 가중치 (0이면 사용 안 함)

    Returns:
        스칼라 loss 텐서 또는 None
    """
    if meta_list is None or lambda_consistency <= 0.0:
        return None

    # 1) group_id -> index 리스트 매핑 생성
    group_map: dict[str, list[int]] = {}
    for idx, meta in enumerate(meta_list):
        # Developer log: meta가 딕셔너리인지 확인하고 안전하게 처리
        if not isinstance(meta, dict):
            continue
        gid = meta.get("group_id")
        if gid is None:
            continue
        group_map.setdefault(gid, []).append(idx)

    # 2) 두 개 이상 샘플이 모인 그룹만 사용
    groups = [idxs for idxs in group_map.values() if len(idxs) > 1]
    if not groups:
        return None

    # 3) 확률 분포와 마스크 계산
    probs = torch.softmax(logits, dim=-1)              # (B, T, V)
    mask = (target_output != pad_id).unsqueeze(-1)     # (B, T, 1), pad 토큰 제외용
    mask = mask.float()

    total_loss = logits.new_tensor(0.0)
    group_count = 0

    for idxs in groups:
        # (G, T, V), (G, T, 1)
        g_probs = probs[idxs]          # 동치 수식 그룹의 출력 분포
        g_mask = mask[idxs]

        # 그룹 평균 분포 (pad 위치는 제외)
        denom = g_mask.sum(dim=0, keepdim=True).clamp_min(1.0)  # (1, T, 1)
        mean_probs = (g_probs * g_mask).sum(dim=0, keepdim=True) / denom  # (1, T, V)

        # 각 샘플 분포가 mean_probs 와 비슷해지도록 L2 penalty
        diff = (g_probs - mean_probs) ** 2
        diff = diff * g_mask  # pad 위치 제외
        group_loss = diff.sum() / g_mask.sum().clamp_min(1.0)

        total_loss = total_loss + group_loss
        group_count += 1

    if group_count == 0:
        return None

    return lambda_consistency * (total_loss / group_count)


def _pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs) if seqs else 1
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


def infix_to_rpn(expr: str) -> List[str]:
    """
    Convert infix expression to RPN (Reverse Polish Notation).
    Developer log: Returns char-level tokens to avoid vocab issues.
    Numbers are split into individual digits for char-level tokenizer.
    
    Example: "12+34" -> ["1", "2", "3", "4", "+"]
    Example: "5*6" -> ["5", "6", "*"]
    Example: "10//2" -> ["1", "0", "2", "D"] (D represents //)
    """
    expr = expr.replace(" ", "")
    output: List[str] = []
    stack: List[str] = []

    def precedence(op: str) -> int:
        if op in ("*", "D"):  # D represents //
            return 2
        if op in ("+", "-"):
            return 1
        return 0

    i = 0
    while i < len(expr):
        ch = expr[i]

        if ch.isdigit():
            # Extract single digit as single char (char-level tokenizer)
            output.append(ch)
            i += 1
            continue

        if ch in "+-*":
            op = ch
            while stack and stack[-1] not in "(" and precedence(stack[-1]) >= precedence(op):
                output.append(stack.pop())
            stack.append(op)
        elif ch == "/":
            # Check for "//" (몫 연산)
            if i + 1 < len(expr) and expr[i + 1] == "/":
                # Use "D" as placeholder for "//" to avoid multi-char issues
                while stack and stack[-1] not in "(" and precedence(stack[-1]) >= precedence("D"):
                    output.append(stack.pop())
                stack.append("D")
                i += 1  # Skip second "/"
        elif ch == "(":
            stack.append(ch)
        elif ch == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
        else:
            # Unknown token → skip
            pass
        i += 1

    while stack:
        token = stack.pop()
        if token != "(":
            output.append(token)

    return output if output else [ch for ch in expr if ch.isdigit() or ch in "+-*"]


def _safe_infix_to_rpn(expr: str) -> List[str]:
    try:
        tokens = infix_to_rpn(expr)
        if tokens:
            return tokens
    except Exception:
        pass
    return [expr]

sweep_config = {
    "method": "random",  # "random", "grid", "bayes" 중 선택
    
    "metric": {
        "name": "valid/EM",
        "goal": "maximize",
    },
    
    "parameters": {
        # Learning rate (증강 데이터 fine-tuning용으로 낮춤)
        "lr": {
            "values": [1.5e-4, 2e-4, 2.5e-4],  # Plan: 1.5e-4 ~ 2.5e-4 범위 탐색
        },
        
        # Model architecture (W&B sweep 최적값 중심으로 확장 탐색)
        "d_model": {
            "values": [256, 384, 512],  # 256이 최적값
        },
        
        # Attention heads (W&B sweep 최적값 중심으로 확장 탐색)
        "nhead": {
            "values": [2, 4, 8],  # 2가 최적값 (256, 384, 512 모두와 호환)
        },
        
        # Encoder/Decoder layers (W&B sweep 최적값: 6/2)
        "num_encoder_layers": {
            "values": [4, 6, 8],  # 6이 최적값
        },
        
        "num_decoder_layers": {
            "values": [2, 4, 6],  # 2가 최적값
        },
        
        # FFN dimension (W&B sweep 최적값 중심으로 확장 탐색)
        "dim_feedforward": {
            "values": [1024, 1536, 2048],  # 1024가 최적값
        },
        
        # Dropout (W&B sweep 최적값: 0.0)
        "dropout": {
            "values": [0.0, 0.1, 0.2],  # 0.0이 최적값
        },
        
        # Batch size (W&B sweep 최적값: 128)
        "batch_size": {
            "values": [128, 256],  # 128/256 두 가지 배치 크기만 탐색
        },
        
        # Data phase (W&B sweep 최적값: phase 2-3에 해당)
        "phase": {
            "values": [2, 3, 4],  # max_depth 2-3에 해당
        },
        # Depth profile sweep: baseline vs legacy_powerup vs deep_context
        "depth_profile": {
            "values": ["baseline", "legacy_powerup", "deep_context"],
        },
        # Developer log: max_train_steps sweep (90k 중심)
        "max_train_steps": {
            "values": [90_000, 120_000],
        },
    },
}

# ======================================================================================

# 1. 학습 루프

# ======================================================================================

def train_loop(

    model: nn.Module,           # 학습할 모델

    dataloader: DataLoader,     # DataLoader를 직접 전달받아 사용합니다.

    input_tokenizer: CharTokenizer,      # (tokenize_batch 유틸리티를 위해 유지)

    output_tokenizer: CharTokenizer,     # 출력 문자 토크나이저

    device: torch.device,       # cpu 또는 cuda

    val_dataloader: DataLoader | None = None,  # 별도 검증 DataLoader (None이면 훈련 배치로 검증)

    *,

    train_config: TrainConfig,  # 학습 설정

    model_config: ModelConfig,  # 모델 설정

    tokenizer_config: TokenizerConfig,  # 토크나이저 설정

    rpn_tokenizer: Optional[CharTokenizer] = None,

):

    # 모델을 GPU/CPU로 보냄

    model.to(device)

    # 옵티마이저: AdamW with weight decay (리뷰 반영)
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    
    # Learning rate scheduler: warmup + cosine decay / optional plateau (리뷰 반영)
    scheduler = None
    scheduler_requires_metric = False  # ReduceLROnPlateau 여부 플래그
    if train_config.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optim,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=train_config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optim,
            T_max=max(
                1,
                (train_config.max_train_steps or 100000) - train_config.warmup_steps,
            ),
            eta_min=train_config.min_lr_threshold,
        )
        scheduler = SequentialLR(
            optim,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_config.warmup_steps],
        )
    elif getattr(train_config, "use_plateau_schedule", False):
        # Developer log: Validation EM 기반 ReduceLROnPlateau 스케줄러
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optim,
            mode="max",
            factor=getattr(train_config, "plateau_factor", 0.5),
            patience=getattr(train_config, "plateau_patience", 2),
            min_lr=train_config.min_lr_threshold,
            verbose=False,
        )
        scheduler_requires_metric = True

    # seq2seq에서 흔히 쓰는 CE loss
    # Developer log: Added label smoothing for better generalization
    # pad 토큰은 무시하도록(ignore_index) 설정

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_tokenizer.pad_id,
        label_smoothing=0.1  # Label smoothing for regularization
    )
    use_rpn_head = (
        rpn_tokenizer is not None
        and hasattr(model, "forward_with_rpn")
        and train_config.lambda_rpn > 0
    )
    loss_fn_rpn = (
        nn.CrossEntropyLoss(ignore_index=rpn_tokenizer.pad_id)
        if use_rpn_head and rpn_tokenizer is not None
        else None
    )
    # Developer log: RPN stack-value regression loss (scratchpad supervision)
    use_rpn_value_head = use_rpn_head and getattr(train_config, "lambda_rpn_value", 0.0) > 0
    loss_fn_rpn_value = nn.MSELoss(reduction="sum") if use_rpn_value_head else None

    step = 0

    model.train()  # 학습 모드로 전환 (Dropout 등 켜짐)

    # tqdm은 진행 상황을 예쁘게 보여주는 라이브러리입니다.

    pbar = tqdm(total=train_config.max_train_steps if train_config.max_train_steps is not None else None, desc="train", unit="step", ncols=120, dynamic_ncols=True, leave=True)

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")
    
    # Early stopping 관련 변수 (wandb sweep용)
    no_improvement_count = 0  # 개선 없는 validation 횟수
    last_improvement_step = 0  # 마지막 개선 step
    initial_lr = train_config.lr  # 초기 학습률

    for epoch in range(train_config.num_epochs):

        # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.

        if train_config.max_train_steps is not None and step >= train_config.max_train_steps: break

        pbar.write(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")

        # 실제로 배치를 하나씩 뽑아서 학습하는 부분입니다.

        for batch in dataloader:

            # --------------------------------------------------------------

            # 1) 토크나이즈 & 텐서로 변환

            #    `tokenize_batch`는 BatchTensors(src, tgt_inp, tgt_out)를 반환합니다.

            #    변수 역할:

            #      - `src` (encoder input): 모델의 인코더 입력. 정수 텐서, shape (B, S).

            #      - `target_input` (tgt_inp): 디코더에 teacher-forcing으로 넣는 입력. shape (B, T).

            #          일반적으로 BOS를 앞에 붙이고 EOS는 제외한 시퀀스입니다.

            #      - `target_output` (tgt_out): 디코더가 예측해야 하는 정답(손실 대상). shape (B, T).

            #          일반적으로 target_input에서 BOS를 뺀 것에 EOS를 붙인 형태입니다.

            #    예시 (토큰 id가 다음과 같다고 가정):

            #      bos_id=1, eos_id=2, '1'->5, '6'->6

            #      원본 target_text: "16"

            #      target_input ids:  [1, 5, 6]    # [BOS, '1', '6']

            #      target_output ids: [5, 6, 2]    # ['1', '6', EOS']

            #    주의: 모든 텐서는 dtype=torch.long이고 `.to(device)`로 명시적 이동이 필요합니다.

            # --------------------------------------------------------------

            batch_tensors = tokenize_batch(batch, input_tokenizer, output_tokenizer)

            src = batch_tensors.src.to(device)

            target_input = batch_tensors.tgt_inp.to(device)

            target_output = batch_tensors.tgt_out.to(device)

            # Forward: 모델에 입력을 전달하고 출력을 얻습니다.

            # 출력은 (B, T, V) 형태로 반환됩니다.

            rpn_inp = rpn_out = None
            result_logits = None
            rpn_logits = None
            rpn_value_pred = None

            if use_rpn_head and rpn_tokenizer is not None and loss_fn_rpn is not None:
                rpn_inp_ids: List[List[int]] = []
                rpn_out_ids: List[List[int]] = []
                rpn_value_targets_list: List[List[float]] = []
                rpn_value_masks_list: List[List[int]] = []
                for idx, expr in enumerate(batch["input_text"]):
                    tokens = _safe_infix_to_rpn(expr)
                    rpn_text = " ".join(tokens)
                    
                    # Debug first batch only
                    if step == 0 and idx == 0:
                        print(f"\n🔍 First RPN example:")
                        print(f"  Input expr: {expr}")
                        print(f"  RPN tokens: {tokens}")
                        print(f"  RPN text: '{rpn_text}'")
                        print(f"  RPN vocab has all chars: {all(ch in rpn_tokenizer.stoi for ch in rpn_text)}")
                        missing = [ch for ch in rpn_text if ch not in rpn_tokenizer.stoi]
                        if missing:
                            print(f"  ❌ Missing chars: {missing}")
                    
                    ids_body = _encode_rpn_text(rpn_tokenizer, rpn_text)
                    
                    if step == 0 and idx == 0:
                        print(f"  Encoded IDs: {ids_body}")
                        print(f"  Max ID: {max(ids_body) if ids_body else 'N/A'}, RPN vocab size: {rpn_tokenizer.vocab_size}")
                        print(f"  All IDs valid: {all(0 <= i < rpn_tokenizer.vocab_size for i in ids_body)}\n")
                    
                    rpn_inp_ids.append([rpn_tokenizer.bos_id] + ids_body)
                    rpn_out_ids.append(ids_body + [rpn_tokenizer.eos_id])

                    # RPN stack-value targets (연산자 위치 기준 log-scale 값)
                    if use_rpn_value_head:
                        stack_vals = _compute_rpn_stack_values(expr)
                        value_seq: List[float] = []
                        mask_seq: List[int] = []
                        op_index = 0
                        for tok in tokens:
                            if tok in {"+", "-", "*", "D"}:
                                if op_index < len(stack_vals):
                                    value_seq.append(stack_vals[op_index])
                                    mask_seq.append(1)
                                    op_index += 1
                                else:
                                    # 방어적: label 부족 시 마스크만 0으로 설정
                                    value_seq.append(0.0)
                                    mask_seq.append(0)
                            else:
                                value_seq.append(0.0)
                                mask_seq.append(0)
                        rpn_value_targets_list.append(value_seq)
                        rpn_value_masks_list.append(mask_seq)

                rpn_inp = _pad_sequences(rpn_inp_ids, rpn_tokenizer.pad_id).to(device)
                rpn_out = _pad_sequences(rpn_out_ids, rpn_tokenizer.pad_id).to(device)
                rpn_value_targets = None
                rpn_value_mask = None
                if use_rpn_value_head and rpn_value_targets_list:
                    max_len_val = max(len(v) for v in rpn_value_targets_list)
                    B_val = len(rpn_value_targets_list)
                    rpn_value_targets = torch.zeros(
                        (B_val, max_len_val), dtype=torch.float32, device=device
                    )
                    rpn_value_mask = torch.zeros(
                        (B_val, max_len_val), dtype=torch.float32, device=device
                    )
                    for bi, (vals, mask_seq) in enumerate(
                        zip(rpn_value_targets_list, rpn_value_masks_list)
                    ):
                        L = len(vals)
                        rpn_value_targets[bi, :L] = torch.tensor(vals, dtype=torch.float32, device=device)
                        rpn_value_mask[bi, :L] = torch.tensor(mask_seq, dtype=torch.float32, device=device)
                
                # Additional safety check
                if step == 0:
                    max_inp_val = rpn_inp.max().item()
                    max_out_val = rpn_out.max().item()
                    print(f"🔍 RPN tensor check:")
                    print(f"  rpn_inp max value: {max_inp_val}, vocab size: {rpn_tokenizer.vocab_size}")
                    print(f"  rpn_out max value: {max_out_val}, vocab size: {rpn_tokenizer.vocab_size}")
                    if max_inp_val >= rpn_tokenizer.vocab_size:
                        print(f"  ❌ ERROR: rpn_inp has index {max_inp_val} >= vocab size {rpn_tokenizer.vocab_size}")
                    if max_out_val >= rpn_tokenizer.vocab_size:
                        print(f"  ❌ ERROR: rpn_out has index {max_out_val} >= vocab size {rpn_tokenizer.vocab_size}")

                result_logits, rpn_logits, rpn_value_pred = model.forward_with_rpn(
                    src=src,
                    tgt_result_inp=target_input,
                    src_pad_id=input_tokenizer.pad_id,
                    rpn_inp=rpn_inp,
                )
            else:
                result_logits = model(src, target_input, input_tokenizer.pad_id)

            # --------------------------------------------------------------

            # 4) Loss 계산

            # --------------------------------------------------------------

            loss = loss_fn(

                result_logits.view(-1, result_logits.size(-1)),  # (B*T, V)

                target_output.view(-1),             # (B*T,)

            )

            if (
                use_rpn_head
                and rpn_logits is not None
                and rpn_out is not None
                and train_config.lambda_rpn > 0
            ):
                loss_rpn = loss_fn_rpn(

                    rpn_logits.view(-1, rpn_logits.size(-1)),

                    rpn_out.view(-1),

                )
                loss = loss + train_config.lambda_rpn * loss_rpn
            else:
                loss_rpn = None

            # RPN stack-value regression loss (scratchpad)
            if (
                use_rpn_value_head
                and loss_fn_rpn_value is not None
                and rpn_value_pred is not None
                and rpn_value_targets is not None
                and rpn_value_mask is not None
            ):
                # rpn_value_pred: [B, T_rpn], rpn_value_targets/mask: [B, T_rpn]
                min_len = min(
                    rpn_value_pred.size(1),
                    rpn_value_targets.size(1),
                )
                pred_trim = rpn_value_pred[:, :min_len]
                target_trim = rpn_value_targets[:, :min_len]
                mask_trim = rpn_value_mask[:, :min_len]
                diff_sq = (pred_trim - target_trim) ** 2
                diff_sq = diff_sq * mask_trim
                denom = mask_trim.sum().clamp_min(1.0)
                loss_rpn_value = loss_fn_rpn_value(diff_sq) / denom
                loss = loss + train_config.lambda_rpn_value * loss_rpn_value
            else:
                loss_rpn_value = None

            # EC Consistency Loss (동치 수식 그룹용)
            batch_meta = batch.get("meta") if isinstance(batch, dict) else None
            loss_ec = compute_ec_consistency_loss(
                logits=result_logits,
                target_output=target_output,
                meta_list=batch_meta,
                pad_id=output_tokenizer.pad_id,
                lambda_consistency=train_config.lambda_ec,
            )
            if loss_ec is not None:
                loss = loss + loss_ec

            # --------------------------------------------------------------

            # 5) Backward + optimizer step

            # --------------------------------------------------------------

            loss.backward()

            # Gradient clipping (리뷰 반영: grad_clip 파라미터 사용)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

            optim.step()
            
            # Learning rate scheduling (리뷰 반영)
            if scheduler is not None and not scheduler_requires_metric:
                scheduler.step()

            optim.zero_grad()

            step += 1

            # Log learning rate if scheduler is used
            current_lr = optim.param_groups[0]['lr']
            log_payload = {
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "step": step,
            }
            if loss_rpn is not None:
                log_payload["train/loss_rpn"] = loss_rpn.item()
            if 'loss_rpn_value' in locals() and loss_rpn_value is not None:
                log_payload["train/loss_rpn_value"] = float(loss_rpn_value.item())
            if 'loss_ec' in locals() and loss_ec is not None:
                log_payload["train/loss_ec"] = float(loss_ec.item())
            wandb.log(log_payload)

            # --------------------------------------------------------------

            # 6) Validation

            # --------------------------------------------------------------

            if step % train_config.valid_every == 0:

                model.eval()  # 평가 모드

                # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.

                with torch.no_grad():

                    preds_all: List[str] = []
                    
                    targets_all: List[str] = []
                    
                    inputs_all: List[str] = [] # 검증 데이터셋의 입력, 정답, 예측 결과를 저장할 리스트
                    metas_all: List[dict] = []  # 카테고리/phase 등 메타 정보 수집용

                    for val_batch in val_dataloader: # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.

                        val_bt = tokenize_batch(val_batch, input_tokenizer, output_tokenizer)

                        val_src = val_bt.src.to(device)

                        gen_ids = model.generate(

                            src=val_src,

                            max_len=train_config.max_gen_len,

                            bos_id=output_tokenizer.bos_id,

                            eos_id=output_tokenizer.eos_id,

                            src_pad_id=input_tokenizer.pad_id,

                        )

                        for i in range(gen_ids.size(0)):

                            seq_chars: List[str] = []

                            for t in gen_ids[i].tolist():

                                idx = int(t)

                                if idx == output_tokenizer.eos_id:

                                    break

                                if idx in output_tokenizer.itos:

                                    ch = output_tokenizer.itos[idx]

                                    if ch.isdigit() or (ch == '-' and not seq_chars):

                                        seq_chars.append(ch)

                            pred_str = "".join(seq_chars)

                            if pred_str == "-":

                                pred_str = ""

                            preds_all.append(pred_str)

                        # 검증 데이터셋의 정답, 입력, 메타를 리스트에 추가합니다.
                        
                        targets_all.extend(val_batch["target_text"])
                        
                        inputs_all.extend(val_batch["input_text"])
                        if "meta" in val_batch:
                            # Developer log: meta가 딕셔너리 리스트인지 확인하고 안전하게 처리
                            meta_list = val_batch["meta"]
                            for meta_item in meta_list:
                                if isinstance(meta_item, dict):
                                    metas_all.append(meta_item)
                                else:
                                    # meta가 문자열이거나 다른 타입인 경우 빈 딕셔너리로 처리
                                    metas_all.append({})
                        else:
                            metas_all.extend({} for _ in val_batch["input_text"])

                    # 검증 데이터셋의 예측, 정답을 사용하여 성능 지표를 계산합니다.
                    em_batch = compute_metrics(preds_all, targets_all)
                    current_em = float(em_batch.get("EM", -1.0))
                    current_lr = optim.param_groups[0]['lr']

                    wandb.log(
                        {
                            "valid/EM": em_batch["EM"],
                            "valid/TES": em_batch["TES"],
                            "step": step,
                        }
                    )

                    # Category-wise EM/TES 로깅 (confusion-style 분석용)
                    if metas_all:
                        category_indices: dict[str, List[int]] = {}
                        for idx, meta in enumerate(metas_all):
                            # Developer log: meta가 딕셔너리인지 확인하고 안전하게 처리
                            if isinstance(meta, dict):
                                cat = meta.get("category", "unknown")
                            else:
                                cat = "unknown"
                            # Developer log: "unknown" 카테고리는 로깅에서 제외 (meta 정보 없는 샘플)
                            if cat != "unknown":
                                category_indices.setdefault(cat, []).append(idx)
                        
                        cat_log_payload: dict[str, float] = {}
                        for cat, idxs in category_indices.items():
                            if not idxs:
                                continue
                            cat_preds = [preds_all[i] for i in idxs]
                            cat_targets = [targets_all[i] for i in idxs]
                            cat_metrics = compute_metrics(cat_preds, cat_targets)
                            cat_em = float(cat_metrics.get("EM", 0.0))
                            cat_tes = float(cat_metrics.get("TES", 0.0))
                            key_prefix = f"valid/{cat}"
                            cat_log_payload[f"{key_prefix}/EM"] = cat_em
                            cat_log_payload[f"{key_prefix}/TES"] = cat_tes
                        if cat_log_payload:
                            cat_log_payload["step"] = step
                            wandb.log(cat_log_payload)

                    # Plateau 스케줄러 사용 시 validation metric 기반으로 step 호출
                    if scheduler is not None and scheduler_requires_metric:
                        scheduler.step(current_em)

                    # 진행바에도 성능을 표시합니다.
                    pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f} LR={current_lr:.2e}")

                    pbar.set_postfix(
                        EM=f"{em_batch['EM']:.3f}",
                        TES=f"{em_batch['TES']:.3f}",
                    )

                    pbar.refresh()

                    # Developer log: best_em 업데이트는 early stopping 활성화 여부와 관계없이 항상 수행
                    # 최고 성능 갱신 체크 및 best model 저장
                    if current_em > best_em:
                        best_em = current_em
                        last_improvement_step = step
                        
                        # 최고 성능 갱신 시 전체 체크포인트 저장
                        if train_config.save_best_path is not None:
                            # 세 config를 dict로 변환하여 저장
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,  # 학습 설정 저장
                                "model_config": model_config.__dict__,  # 모델 설정 저장
                                "tokenizer_config": tokenizer_config.__dict__,  # 토크나이저 설정 저장
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            pbar.write(f"New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")
                    
                    # Early stopping 체크 (wandb sweep용)
                    should_stop = False
                    stop_reason = ""
                    
                    if train_config.enable_early_stopping:
                        # EM 개선 체크 (best_em은 이미 위에서 업데이트되었으므로, 
                        # current_em == best_em이면 개선된 것)
                        if current_em == best_em and current_em > float("-inf"):
                            # 개선된 경우 early stopping 카운터 리셋
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        # 1. 학습률이 너무 낮아졌는지 체크
                        if current_lr < train_config.min_lr_threshold:
                            should_stop = True
                            stop_reason = f"Learning rate too low: {current_lr:.2e} < {train_config.min_lr_threshold:.2e}"
                        
                        # 2. Patience 동안 개선이 없고, EM이 최소 임계값 이하인 경우
                        elif (no_improvement_count >= train_config.early_stopping_patience and 
                              current_em < train_config.min_em_threshold):
                            should_stop = True
                            stop_reason = (f"No improvement for {no_improvement_count} validations "
                                         f"(EM={current_em:.3f} < {train_config.min_em_threshold:.3f})")
                        
                        # 3. Patience 동안 개선이 없고, 충분한 step을 학습한 경우
                        elif (no_improvement_count >= train_config.early_stopping_patience and 
                              step >= train_config.warmup_steps + 5000):  # 최소 warmup + 5k step은 학습
                            should_stop = True
                            stop_reason = (f"No improvement for {no_improvement_count} validations "
                                         f"after {step} steps (best EM: {best_em:.3f})")
                        
                        if should_stop:
                            pbar.write("=" * 80)
                            pbar.write(f"⚠️  Early stopping triggered: {stop_reason}")
                            pbar.write(f"   Best EM: {best_em:.3f} at step {last_improvement_step}")
                            pbar.write(f"   Current EM: {current_em:.3f}, LR: {current_lr:.2e}")
                            pbar.write("=" * 80)
                            
                            # wandb에 early stopping 정보 로깅
                            wandb.log({
                                "early_stopping/triggered": True,
                                "early_stopping/reason": stop_reason,
                                "early_stopping/best_em": best_em,
                                "early_stopping/step": step,
                            })
                            
                            # wandb run 종료하여 다음 파라미터 조합으로 넘어가기
                            try:
                                wandb.finish()
                            except:
                                pass
                            
                            return  # train_loop 종료

                    # Developer log: 카테고리별로 다양한 validation 샘플 10개 선택 (ec-focus 개선 반영)
                    pbar.write("=" * 80)
                    pbar.write("Sample Validation Output (카테고리별 다양하게 10개):")
                    pbar.write("=" * 80)
                    
                    # 카테고리별로 샘플 분류
                    import re
                    categorized_samples = {
                        "division": [],      # 나눗셈
                        "subtraction": [],   # 뺄셈
                        "addition": [],      # 덧셈
                        "multiplication": [], # 곱셈
                        "mixed": [],         # 혼합연산
                        "parentheses": [],   # 괄호
                        "large_number": [],  # 큰 수 (5자리+)
                        "identity": [],      # 항등원 (0, 1)
                        "op3_plus": [],      # 연산자 3개 이상
                    }
                    
                    for i, (inp, tgt, pred) in enumerate(zip(inputs_all, targets_all, preds_all)):
                        # 카테고리 판별
                        has_paren = "(" in inp
                        has_div = "//" in inp
                        has_sub = "-" in inp and not inp.startswith("-")
                        has_add = "+" in inp
                        has_mul = "*" in inp and not has_div
                        result_large = len(tgt) >= 5
                        
                        # 연산자 개수
                        op_count = inp.count('+') + inp.count('-') + inp.count('*') + inp.count('//')
                        
                        # 우선순위로 분류 (각 카테고리당 1개씩)
                        if has_paren and len(categorized_samples["parentheses"]) < 1:
                            categorized_samples["parentheses"].append((i, inp, tgt, pred, "괄호"))
                        elif result_large and len(categorized_samples["large_number"]) < 1:
                            categorized_samples["large_number"].append((i, inp, tgt, pred, "큰수(5+자리)"))
                        elif ("+0" in inp or "0+" in inp or "*1" in inp or "1*" in inp) and len(categorized_samples["identity"]) < 1:
                            categorized_samples["identity"].append((i, inp, tgt, pred, "항등원"))
                        elif op_count >= 3 and len(categorized_samples["op3_plus"]) < 1:
                            categorized_samples["op3_plus"].append((i, inp, tgt, pred, "연산자3+"))
                        elif has_div and len(categorized_samples["division"]) < 1:
                            categorized_samples["division"].append((i, inp, tgt, pred, "나눗셈"))
                        elif has_sub and not has_add and not has_mul and len(categorized_samples["subtraction"]) < 1:
                            categorized_samples["subtraction"].append((i, inp, tgt, pred, "뺄셈"))
                        elif has_add and not has_sub and not has_mul and not has_div and len(categorized_samples["addition"]) < 1:
                            categorized_samples["addition"].append((i, inp, tgt, pred, "덧셈"))
                        elif has_mul and not has_add and not has_sub and not has_div and len(categorized_samples["multiplication"]) < 1:
                            categorized_samples["multiplication"].append((i, inp, tgt, pred, "곱셈"))
                        elif op_count > 1 and len(categorized_samples["mixed"]) < 1:
                            categorized_samples["mixed"].append((i, inp, tgt, pred, "혼합"))
                    
                    # 10개 샘플 선택 (각 카테고리에서 1개씩)
                    priority_categories = [
                        "parentheses", "op3_plus", "large_number", "mixed",
                        "identity", "division", "subtraction", "multiplication",
                        "addition",
                    ]
                    
                    selected_samples = []
                    for cat in priority_categories:
                        if categorized_samples[cat]:
                            selected_samples.append(categorized_samples[cat][0])
                        if len(selected_samples) >= train_config.show_valid_samples:
                            break
                    
                    # 부족하면 앞에서 채우기
                    if len(selected_samples) < train_config.show_valid_samples:
                        for i in range(min(train_config.show_valid_samples, len(inputs_all))):
                            if not any(s[0] == i for s in selected_samples):
                                selected_samples.append((i, inputs_all[i], targets_all[i], preds_all[i], "기타"))
                            if len(selected_samples) >= train_config.show_valid_samples:
                                break
                    
                    # 출력
                    for idx, (orig_idx, inp, tgt, pred, label) in enumerate(selected_samples):
                        ok = "✓" if pred == tgt else "✗"
                        
                        # 자리수 정보
                        numbers = re.findall(r'\d+', inp)
                        digit_info = ""
                        if numbers:
                            max_digits = max(len(n) for n in numbers)
                            num_count = len(numbers)
                            digit_info = f"{num_count}n{max_digits}d"
                        
                        pbar.write(f"  [{idx:2d}] {ok} [{label:15s}] {digit_info:7s} | "
                                 f"in: {inp:28s} | tgt: {tgt:9s} | pred: {pred:9s}")
                    
                    pbar.write("=" * 80)

                model.train()  # 다시 학습 모드로

            # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.

            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:

                break # 학습을 종료합니다.

            pbar.update(1) # tqdm 진행 1 step

# ======================================================================================

# 2. main 함수

# ======================================================================================

def main():

    # Wandb 초기화
    wandb.init(
        project="inthon-2025-arithmetic",
        name="checkpoint-resume-training",
        config={
            "mode": "checkpoint_resume",
            "augmentation": True,
            "phase": 2,
        }
    )

    # GPU가 있으면 GPU, 없으면 CPU 사용

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------

    # 1) 데이터 준비

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------

    # 2) 토크나이저 설정 준비

    # --------------------------------------------------------------------------

    # 토크나이저 설정 (별도로 관리)

    tokenizer_config = TokenizerConfig(

        input_chars=INPUT_CHARS,

        output_chars=OUTPUT_CHARS,

        add_special=True,

    )

    # --------------------------------------------------------------------------

    # 3) 토크나이저 준비

    # --------------------------------------------------------------------------

    # 입력 문자 토크나이저, 출력 문자 토크나이저를 준비합니다. 자세한 설정은 model.py를 참고하세요.

    input_tokenizer = CharTokenizer(

        tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,

        add_special=tokenizer_config.add_special,

    )

    output_tokenizer = CharTokenizer(

        tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,

        add_special=tokenizer_config.add_special,

    )
    
    rpn_tokenizer = build_rpn_tokenizer(tokenizer_config)
    
    # Debug RPN tokenizer vocab
    print("\n" + "=" * 70)
    print("🔍 RPN Tokenizer Debugging Info")
    print("=" * 70)
    print(f"RPN vocab size: {rpn_tokenizer.vocab_size}")
    print(f"RPN vocab chars: {sorted(rpn_tokenizer.stoi.keys())}")
    print(f"OUTPUT_CHARS: {list(OUTPUT_CHARS)}")
    print(f"RPN_EXTRA_CHARS: {RPN_EXTRA_CHARS}")
    print("=" * 70 + "\n")

    # --------------------------------------------------------------------------

    # 4) 모델 설정 준비

    # --------------------------------------------------------------------------

    # 모델 아키텍처 설정 (별도로 관리)

    # model_config = ModelConfig(

    #   d_model=256,

    #   n_head = 4,

    #   num_encoder_layers = 4,

    #   num_decoder_layers  = 4,

    #   dim_feedforward  = 512,

    #   dropout = 0.1, )

    # # --------------------------------------------------------------------------

    # # 5) 학습 설정 준비

    # # --------------------------------------------------------------------------

    # # 학습 하이퍼파라미터 설정

    # train_config = TrainConfig(

    #     max_train_steps=None,

    #     lr=2e-3,

    #     valid_every=200,

    #     max_gen_len=24,

    #     show_valid_samples=5,

    #     num_epochs=10,

    #     save_best_path="best_model.pt",

    # )

    #-----------------------------

    # Weight & Biases 모델 설정 준비

    #-----------------------------

    base_model_config = ModelConfig()

    train_config = TrainConfig(
        save_best_path="best_model.pt",
        enable_early_stopping=False,  # 전체 학습 진행
    )

    model_config = apply_depth_profile(base_model_config, train_config.depth_profile)

    # --------------------------------------------------------------------------
    #
    # 1-2) 데이터 준비 (Train / Validation)
    #
    # --------------------------------------------------------------------------
    
    # Developer log: EC enhanced data generation strategy
    # 1) Base 데이터 (40만개) - 연산자 1-4개 고르게 분포
    print("=" * 70)
    print("📊 Generating Base Dataset (400k samples)")
    print("=" * 70)
    
    base_dataset = ArithmeticDataset(
        num_samples=400_000,
        phase=train_config.train_phase_mix[0],
        phase_mix=train_config.train_phase_mix,
        seed=123,
        mode="train",
        enable_augmentation=False,  # 원본만 생성
    )
    
    # 2) 증강 데이터 생성 (40만 → 60만개)
    # Developer log: Expression pairs with group_id for EC learning
    print("\n" + "=" * 70)
    print("🔄 Generating Augmented Dataset (40k → 60k with expression pairs)")
    print("=" * 70)
    
    augmented_data_list = create_augmented_dataset_from_original(base_dataset)
    
    # 3) Augmented dataset을 Dataset으로 wrapping
    class AugmentedDatasetWrapper(Dataset):
        """Wrapper for augmented data list with group_id support."""
        def __init__(self, data_list):
            self.data = data_list
            self.mode = "train"
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = AugmentedDatasetWrapper(augmented_data_list)
    
    print(f"\n✅ Final training dataset: {len(train_dataset)} samples")
    print(f"   - Original: ~400k samples")
    print(f"   - Augmented: ~{len(train_dataset) - 400_000} samples")
    print(f"   - Total: {len(train_dataset)} samples with group_id for EC\n")

    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = ArithmeticDataset(
        num_samples=train_config.val_num_samples,
        phase=train_config.val_phase,
        seed=999,
        mode="val",
        enable_augmentation=False,
    )

    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=min(train_config.batch_size, 256),
        num_workers=0,
        pin_memory=True,
        mode="val",
    )

    # --------------------------------------------------------------------------

    # 6) 모델 준비

    # --------------------------------------------------------------------------

    # GRU 기반의 TinySeq2Seq 모델을 준비합니다. 자세한 설정은 model.py를 참고하세요.

    # ModelConfig의 모든 필드를 **kwargs로 전달

    # model = TinySeq2Seq(

    #     in_vocab=input_tokenizer.vocab_size,

    #     out_vocab=output_tokenizer.vocab_size,

    #     **model_config.__dict__,  # 모델 설정을 **kwargs로 전달

    # )

    rpn_vocab = (
        rpn_tokenizer.vocab_size
        if rpn_tokenizer is not None and train_config.lambda_rpn > 0
        else None
    )
    
    print(f"\n🔍 Model initialization:")
    print(f"  in_vocab: {input_tokenizer.vocab_size}")
    print(f"  out_vocab: {output_tokenizer.vocab_size}")
    print(f"  rpn_vocab: {rpn_vocab}")
    print(f"  lambda_rpn: {train_config.lambda_rpn}")
    print()

    model = TransformerSeq2Seq(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        rpn_vocab=rpn_vocab,
        **model_config.__dict__,
    )
    
    # Verify RPN head was initialized correctly
    if rpn_vocab is not None:
        print(f"✅ RPN head initialized:")
        print(f"  rpn_embed num_embeddings: {model.rpn_embed.num_embeddings if model.rpn_embed else 'None'}")
        print(f"  rpn_out out_features: {model.rpn_out.out_features if model.rpn_out else 'None'}")
        print()

    # --------------------------------------------------------------------------
    # 체크포인트 로드 (Resume training with compatibility check)
    # --------------------------------------------------------------------------
    
    resume_checkpoint = "best_model.pt"  # 체크포인트 파일 경로
    resume_from_checkpoint = True  # True로 설정하면 체크포인트에서 재개
    
    # A100 최적화 전 모델 설정 (d_model=256, nhead=2)
    OLD_MODEL_CONFIG = {
        "d_model": 256,
        "nhead": 2,
        "num_encoder_layers": 6,
        "num_decoder_layers": 2,
        "dim_feedforward": 1024,
        "dropout": 0.0,
    }
    
    if resume_from_checkpoint and os.path.exists(resume_checkpoint):
        print("=" * 70)
        print(f"🔄 Checking checkpoint: {resume_checkpoint}")
        print("=" * 70)
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            # 저장된 설정 확인 및 비교
            config_match = True
            if "model_config" in checkpoint:
                saved_config = checkpoint["model_config"]
                print("\n📋 Checkpoint Configuration Comparison:")
                print(f"{'Parameter':<25} {'Current':<12} {'Saved':<12} {'Match'}")
                print("-" * 70)
                
                for key in ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward", "dropout"]:
                    current_val = model_config.__dict__[key]
                    saved_val = saved_config.get(key, "N/A")
                    is_match = current_val == saved_val
                    match_symbol = "✅" if is_match else "❌"
                    print(f"{key:<25} {str(current_val):<12} {str(saved_val):<12} {match_symbol}")
                    if not is_match:
                        config_match = False
            
            state_dict = checkpoint["model_state"]
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing_keys = incompatible.missing_keys
            unexpected_keys = incompatible.unexpected_keys

            if config_match:
                print("\n✅ Configuration matches! Model weights loaded successfully.")
            else:
                print("\n⚠️  Model configuration mismatch detected!")
                print("   Current model depth differs from checkpoint (expected for depth_profile).")
                print("   Missing keys will be randomly initialized; training will fine-tune them.")

            if missing_keys:
                print(f"   ℹ️ Missing keys ({len(missing_keys)}): {missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}")
            if unexpected_keys:
                print(f"   ℹ️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:8]}{' ...' if len(unexpected_keys) > 8 else ''}")

            # Verify RPN head after checkpoint loading
            if rpn_vocab is not None and model.rpn_embed is not None:
                actual_rpn_embed_size = model.rpn_embed.num_embeddings
                if actual_rpn_embed_size != rpn_vocab:
                    print(f"   ⚠️ WARNING: RPN embed size mismatch!")
                    print(f"      Expected: {rpn_vocab}, Got: {actual_rpn_embed_size}")
                    print(f"      This will cause CUDA index errors. Reinitializing RPN head...")
                    # Reinitialize RPN head with correct size
                    d_model = model_config.d_model
                    nhead = model_config.nhead
                    dim_feedforward = model_config.dim_feedforward
                    dropout = model_config.dropout
                    num_decoder_layers = model_config.num_decoder_layers
                    
                    rpn_decoder_layer = nn.TransformerDecoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True,
                    )
                    model.rpn_decoder = nn.TransformerDecoder(rpn_decoder_layer, num_layers=num_decoder_layers)
                    model.rpn_embed = nn.Embedding(rpn_vocab, d_model)
                    model.rpn_out = nn.Linear(d_model, rpn_vocab)
                    model.to(device)  # Move new layers to device
                    print(f"      ✅ RPN head reinitialized with vocab size {rpn_vocab}")

            if "step" in checkpoint:
                print(f"📊 Resuming from step: {checkpoint['step']}")
            
            print("=" * 70)
            print()
            
        except Exception as e:
            print(f"\n❌ Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
            print("=" * 70)
            print()
    
    elif resume_from_checkpoint and not os.path.exists(resume_checkpoint):
        print("=" * 70)
        print(f"ℹ️  Checkpoint file not found: {resume_checkpoint}")
        print("🚀 Starting training from scratch with A100-optimized model...")
        print("=" * 70)
        print()

    # --------------------------------------------------------------------------

    # 6) 학습 시작

    # --------------------------------------------------------------------------

    train_loop(

        model=model,

        dataloader=train_dataloader,              # 미리 만든 DataLoader를 직접 전달합니다.

        input_tokenizer=input_tokenizer,        # (tokenize_batch 유틸리티를 위해 전달)

        output_tokenizer=output_tokenizer,

        device=device,

        val_dataloader=val_dataloader,

        train_config=train_config,  # 학습 설정 전달

        model_config=model_config,  # 모델 설정 전달

        tokenizer_config=tokenizer_config,  # 토크나이저 설정 전달

        rpn_tokenizer=rpn_tokenizer,

    )

    # --------------------------------------------------------------------------

    # 6) 학습 끝나면 모델 저장

    # --------------------------------------------------------------------------

    torch.save(model.state_dict(), "model.pt")

    print("Saved model.pt")

def train_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(project="inthon-2025-arithmetic"):
        cfg = wandb.config
        run_name = wandb.run.name  # finish 이후 접근 불가하므로 미리 저장

        tokenizer_config = TokenizerConfig(
            input_chars=INPUT_CHARS,
            output_chars=OUTPUT_CHARS,
            add_special=True,
        )

        input_tokenizer = CharTokenizer(
            tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,
            add_special=tokenizer_config.add_special,
        )

        output_tokenizer = CharTokenizer(
            tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,
            add_special=tokenizer_config.add_special,
        )
        rpn_tokenizer = build_rpn_tokenizer(tokenizer_config)

        phase_mix_cfg = cfg.get("phase_mix", None)
        if phase_mix_cfg is not None:
            if isinstance(phase_mix_cfg, (list, tuple)):
                phase_mix = tuple(max(1, min(4, int(p))) for p in phase_mix_cfg)
            else:
                phase_mix = (max(1, min(4, int(phase_mix_cfg))),)
        elif cfg.get("phase", None) is not None:
            phase_mix = (max(1, min(4, int(cfg.get("phase", 2)))),)
        else:
            phase_mix = (2, 3, 4)

        train_config = TrainConfig(
            max_train_steps=cfg.get("max_train_steps", None),
            lr=cfg.lr,
            warmup_steps=5000,
            weight_decay=0.1,
            grad_clip=1.0,
            valid_every=200,
            max_gen_len=50,
            show_valid_samples=5,
            num_epochs=20,
            save_best_path=f"best_{run_name}.pt",
            use_cosine_schedule=True,
            enable_early_stopping=True,
            early_stopping_patience=5,
            min_lr_threshold=1e-6,
            min_em_threshold=0.01,
            batch_size=cfg.batch_size,
            depth_profile=cfg.get("depth_profile", "baseline"),
            train_num_samples=cfg.get("train_num_samples", 900_000),
            val_num_samples=cfg.get("val_num_samples", 3_000),
            train_phase_mix=phase_mix,
            val_phase=max(1, min(4, int(cfg.get("val_phase", 4)))),
        )

        override_fields = [
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
        ]
        if all(hasattr(cfg, field) for field in override_fields):
            model_config = ModelConfig(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                num_encoder_layers=cfg.num_encoder_layers,
                num_decoder_layers=cfg.num_decoder_layers,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
            )
        else:
            base_model_config = ModelConfig()
            model_config = apply_depth_profile(base_model_config, train_config.depth_profile)

        # Developer log: EC enhanced data generation (same as main)
        base_dataset = ArithmeticDataset(
            num_samples=400_000,
            phase=train_config.train_phase_mix[0],
            phase_mix=train_config.train_phase_mix,
            seed=123,
            mode="train",
            enable_augmentation=False,
        )
        
        augmented_data_list = create_augmented_dataset_from_original(base_dataset)
        
        class AugmentedDatasetWrapper(Dataset):
            def __init__(self, data_list):
                self.data = data_list
                self.mode = "train"
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        train_dataset = AugmentedDatasetWrapper(augmented_data_list)

        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=train_config.batch_size,
            num_workers=0,
            pin_memory=True,
        )

        val_dataset = ArithmeticDataset(
            num_samples=train_config.val_num_samples,
            phase=train_config.val_phase,
            seed=999,
            mode="val",
            enable_augmentation=False,
        )

        val_dataloader = get_dataloader(
            val_dataset,
            batch_size=min(train_config.batch_size, 256),
            num_workers=0,
            pin_memory=True,
            mode="val",
        )

        rpn_vocab = (
            rpn_tokenizer.vocab_size
            if rpn_tokenizer is not None and train_config.lambda_rpn > 0
            else None
        )

        model = TransformerSeq2Seq(
            in_vocab=input_tokenizer.vocab_size,
            out_vocab=output_tokenizer.vocab_size,
            rpn_vocab=rpn_vocab,
            **model_config.__dict__,
        )

        train_loop(
            model=model,
            dataloader=train_dataloader,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
            device=device,
            val_dataloader=val_dataloader,
            train_config=train_config,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            rpn_tokenizer=rpn_tokenizer,
        )

        torch.save(model.state_dict(), "model_last.pt")
        print("Saved model_last.pt for run:", run_name)

# python train.py로 실행했을 때만 main()을 돌게 합니다.

if __name__ == "__main__":

    main()

    #sweep_id = wandb.sweep(sweep_config, project="inthon-2025-arithmetic")

    #wandb.agent(sweep_id, function=train_run, count=20)  # 20번 실험 (원하는 만큼 조정)
