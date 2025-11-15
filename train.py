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
            "values": [1e-4, 2e-4, 5e-5],  # Fine-tuning: 기존보다 2-5배 낮춤
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
            "values": [64, 128, 256],  # 128이 최적값
        },
        
        # Data phase (W&B sweep 최적값: phase 2-3에 해당)
        "phase": {
            "values": [2, 3, 4],  # max_depth 2-3에 해당
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
    
    # Learning rate scheduler: warmup + cosine decay (리뷰 반영)
    if train_config.use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optim, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=train_config.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optim,
            T_max=max(1, (train_config.max_train_steps or 100000) - train_config.warmup_steps),
            eta_min=1e-6
        )
        scheduler = SequentialLR(
            optim,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_config.warmup_steps]
        )
    else:
        scheduler = None

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

            if use_rpn_head and rpn_tokenizer is not None and loss_fn_rpn is not None:
                rpn_inp_ids: List[List[int]] = []
                rpn_out_ids: List[List[int]] = []
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

                rpn_inp = _pad_sequences(rpn_inp_ids, rpn_tokenizer.pad_id).to(device)
                rpn_out = _pad_sequences(rpn_out_ids, rpn_tokenizer.pad_id).to(device)
                
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

                result_logits, rpn_logits = model.forward_with_rpn(
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

            # --------------------------------------------------------------

            # 5) Backward + optimizer step

            # --------------------------------------------------------------

            loss.backward()

            # Gradient clipping (리뷰 반영: grad_clip 파라미터 사용)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

            optim.step()
            
            # Learning rate scheduling (리뷰 반영)
            if scheduler is not None:
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

                        # 검증 데이터셋의 정답, 입력을 리스트에 추가합니다.

                        targets_all.extend(val_batch["target_text"])

                        inputs_all.extend(val_batch["input_text"])

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

                    # 진행바에도 성능을 표시합니다.
                    pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f} LR={current_lr:.2e}")

                    pbar.set_postfix(
                        EM=f"{em_batch['EM']:.3f}",
                        TES=f"{em_batch['TES']:.3f}",
                    )

                    pbar.refresh()

                    # Early stopping 체크 (wandb sweep용)
                    should_stop = False
                    stop_reason = ""
                    
                    if train_config.enable_early_stopping:
                        # 1. EM 개선 체크
                        if current_em > best_em:
                            best_em = current_em
                            no_improvement_count = 0
                            last_improvement_step = step
                        else:
                            no_improvement_count += 1
                        
                        # 2. 학습률이 너무 낮아졌는지 체크
                        if current_lr < train_config.min_lr_threshold:
                            should_stop = True
                            stop_reason = f"Learning rate too low: {current_lr:.2e} < {train_config.min_lr_threshold:.2e}"
                        
                        # 3. Patience 동안 개선이 없고, EM이 최소 임계값 이하인 경우
                        elif (no_improvement_count >= train_config.early_stopping_patience and 
                              current_em < train_config.min_em_threshold):
                            should_stop = True
                            stop_reason = (f"No improvement for {no_improvement_count} validations "
                                         f"(EM={current_em:.3f} < {train_config.min_em_threshold:.3f})")
                        
                        # 4. Patience 동안 개선이 없고, 충분한 step을 학습한 경우
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
                    
                    # 최고 성능 갱신 시 전체 체크포인트 저장
                    # (best_em은 이미 early stopping 체크에서 업데이트됨)
                    if train_config.save_best_path is not None:
                        # Early stopping에서 이미 best_em이 업데이트되었으므로, 
                        # current_em == best_em인 경우에만 저장
                        if current_em == best_em and current_em > float("-inf"):
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

                    # 고정된 validation 샘플 표시 (원래 방식 유지 + 카테고리 라벨 추가)
                    # Validation 데이터셋이 고정되어 있으므로, 항상 같은 인덱스의 샘플을 보여줌
                    B = len(preds_all)  # 검증 데이터셋의 크기
                    n_show = min(train_config.show_valid_samples, B)
                    
                    pbar.write("=" * 80)
                    pbar.write("Sample Validation Output (대회 평가 기준별):")
                    pbar.write("=" * 80)
                    
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "✓" if pred == tgt else "✗"
                        
                        # 카테고리 라벨 + 자리수 정보 추가
                        import re
                        category_label = ""
                        digit_info = ""
                        
                        # 입력 수식에서 숫자 추출 및 자리수 분석
                        numbers = re.findall(r'\d+', input_str)
                        if numbers:
                            max_digits = max(len(n) for n in numbers)
                            num_count = len(numbers)
                            digit_info = f"{num_count}n{max_digits}d"
                        
                        # 상세 카테고리 분류
                        if "(" in input_str and "*" in input_str and ("+" in input_str or "-" in input_str):
                            category_label = "[Mix:Paren*±]"  # 괄호+혼합
                        elif "(" in input_str:
                            category_label = "[Parentheses]"
                        elif "//" in input_str:
                            category_label = "[Division]"
                        elif "-" in input_str and "+" not in input_str and "*" not in input_str:
                            category_label = "[Subtraction]"
                        elif any(pattern in input_str for pattern in ["+0", "*1", "+1", "*0", "0+", "1*"]):
                            category_label = "[Identity]"
                        elif len(tgt) >= 6:
                            category_label = "[OOD:6+dig]"
                        elif "*" in input_str and "+" not in input_str and "-" not in input_str:
                            category_label = "[Multiply]"
                        elif "+" in input_str and "*" not in input_str and "-" not in input_str:
                            category_label = "[Addition]"
                        elif "*" in input_str and "+" in input_str:
                            category_label = "[Mix:*+]"
                        elif "*" in input_str and "-" in input_str:
                            category_label = "[Mix:*-]"
                        else:
                            category_label = "[Basic]"
                        
                        pbar.write(f"  [{i:2d}] {ok} {category_label:16s} {digit_info:7s} | "
                                 f"in: {input_str:28s} | tgt: {tgt:9s} | pred: {pred:9s}")
                    
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
            max_train_steps=None,
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
