from __future__ import annotations



from typing import List, Any, Tuple

from config import TrainConfig, ModelConfig, TokenizerConfig

import os

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, IterableDataset

from tqdm import tqdm

import wandb

from dataloader import (

    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset

    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수

)

# 고정된 validation 샘플 (전역 변수로 한 번만 생성)
_FIXED_VAL_SAMPLES = None
_FIXED_VAL_SAMPLES_INPUTS = None

from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (

    TinySeq2Seq,

    TransformerSeq2Seq,

    CharTokenizer,      # 문자 단위 토크나이저

    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수

    INPUT_CHARS,        # 입력 문자 집합

    OUTPUT_CHARS,       # 출력 문자 집합

)

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

    # pad 토큰은 무시하도록(ignore_index) 설정

    loss_fn = nn.CrossEntropyLoss(ignore_index=output_tokenizer.pad_id)

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

            logits = model(src, target_input, input_tokenizer.pad_id)

            # --------------------------------------------------------------

            # 4) Loss 계산

            # --------------------------------------------------------------

            loss = loss_fn(

                logits.view(-1, logits.size(-1)),  # (B*T, V)

                target_output.view(-1),             # (B*T,)

            )

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
            wandb.log({
                "train/loss": loss.item(), 
                "train/lr": current_lr,
                "step": step
            })

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

                    # 카테고리별 validation 샘플 10개 선택 및 표시
                    pbar.write("=" * 80)
                    pbar.write("Sample Validation Output (카테고리별 10개):")
                    pbar.write("=" * 80)
                    
                    # 카테고리별로 샘플 분류
                    categorized_samples = {
                        "division": [],      # 나눗셈
                        "subtraction": [],   # 뺄셈
                        "addition": [],      # 덧셈
                        "multiplication": [], # 곱셈
                        "mixed": [],         # 혼합연산
                        "parentheses": [],   # 괄호있는연산
                        "large_number": [],  # 자리수 큰 사칙연산 (5자리+)
                        "commutative": [],   # 교환법칙 (A+B, A*B 형태)
                        "with_zero": [],     # 0포함 연산
                        "with_one": [],      # 1포함 연산
                    }
                    
                    for i, (inp, tgt, pred) in enumerate(zip(inputs_all, targets_all, preds_all)):
                        # 카테고리 판별
                        has_paren = "(" in inp
                        has_div = "//" in inp
                        has_sub = "-" in inp and not inp.startswith("-")
                        has_add = "+" in inp
                        has_mul = "*" in inp and not has_div
                        has_zero = "0" in inp
                        has_one = "1" in inp
                        result_large = len(tgt) >= 5
                        
                        # 연산자 개수
                        op_count = sum([has_div, has_sub, has_add, has_mul])
                        
                        # 우선순위로 분류
                        if has_paren and len(categorized_samples["parentheses"]) < 1:
                            categorized_samples["parentheses"].append((i, inp, tgt, pred, "괄호연산"))
                        elif result_large and len(categorized_samples["large_number"]) < 1:
                            categorized_samples["large_number"].append((i, inp, tgt, pred, "큰수연산(5+자리)"))
                        elif "+0" in inp or "0+" in inp and len(categorized_samples["with_zero"]) < 1:
                            categorized_samples["with_zero"].append((i, inp, tgt, pred, "0포함연산"))
                        elif "*1" in inp or "1*" in inp and len(categorized_samples["with_one"]) < 1:
                            categorized_samples["with_one"].append((i, inp, tgt, pred, "1포함연산"))
                        elif has_div and len(categorized_samples["division"]) < 1:
                            categorized_samples["division"].append((i, inp, tgt, pred, "나눗셈"))
                        elif has_sub and not has_add and not has_mul and len(categorized_samples["subtraction"]) < 1:
                            categorized_samples["subtraction"].append((i, inp, tgt, pred, "뺄셈"))
                        elif has_add and not has_sub and not has_mul and not has_div and len(categorized_samples["addition"]) < 1:
                            categorized_samples["addition"].append((i, inp, tgt, pred, "덧셈"))
                        elif has_mul and not has_add and not has_sub and not has_div and len(categorized_samples["multiplication"]) < 1:
                            categorized_samples["multiplication"].append((i, inp, tgt, pred, "곱셈"))
                        elif op_count > 1 and len(categorized_samples["mixed"]) < 1:
                            categorized_samples["mixed"].append((i, inp, tgt, pred, "혼합연산"))
                        elif (has_add or has_mul) and not has_paren and len(categorized_samples["commutative"]) < 1:
                            categorized_samples["commutative"].append((i, inp, tgt, pred, "교환법칙"))
                    
                    # 10개 샘플 수집 (우선순위 순서)
                    priority_categories = [
                        "commutative", "parentheses", "large_number", "mixed",
                        "addition", "multiplication", "subtraction", "division",
                        "with_zero", "with_one"
                    ]
                    
                    selected_samples = []
                    for cat in priority_categories:
                        if categorized_samples[cat]:
                            selected_samples.append(categorized_samples[cat][0])
                        if len(selected_samples) >= 10:
                            break
                    
                    # 부족하면 앞에서부터 채우기
                    if len(selected_samples) < 10:
                        for i in range(min(10, len(inputs_all))):
                            if not any(s[0] == i for s in selected_samples):
                                selected_samples.append((i, inputs_all[i], targets_all[i], preds_all[i], "기타"))
                            if len(selected_samples) >= 10:
                                break
                    
                    # 출력
                    for idx, (_, inp, tgt, pred, label) in enumerate(selected_samples):
                        ok = "✓" if pred == tgt else "✗"
                        pbar.write(f"  [{idx:2d}] {ok} [{label:15s}] | "
                                 f"input: {inp:30s} | target: {tgt:12s} | pred: {pred:12s}")
                    
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
        name="ec-focus-finetuning",
        config={
            "mode": "ec_focus_finetuning",
            "augmentation": True,
            "phase": 2,
            "ec_ratio": 0.40,  # EC 비중 40%
        }
    )

    # GPU가 있으면 GPU, 없으면 CPU 사용

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------

    # 1) 데이터 준비 (EC 집중)

    # --------------------------------------------------------------------------

    # Train Dataset, 자세한 설정은 dataloader.py를 참고하세요.

    # W&B sweep 최적값 적용 (max_depth 2-3 → phase 2-3)
    train_dataset = ArithmeticDataset(
        num_samples=300_000,  # 샘플 수 증가 (200k → 300k)
        phase=2,  # Phase 2: max_depth_train=2에 해당 (2-3자리)
        seed=123,
        mode="train",
        enable_augmentation=True,  # EC augmentation 활성화
    )

    # Train DataLoader, 자세한 설정은 dataloader.py를 참고하세요.

    train_dataloader = get_dataloader(

        train_dataset,

        batch_size=128,  # W&B sweep 최적값: 128

        num_workers=0,

        pin_memory=True,

    )

    # Validation Dataset: 고정된 validation 샘플 사용
    global _FIXED_VAL_SAMPLES, _FIXED_VAL_SAMPLES_INPUTS
    
    if _FIXED_VAL_SAMPLES is None:
        # 고정된 validation dataset 생성 (W&B sweep 최적값 적용: max_depth_val=3 → phase=3)
        fixed_val_dataset = ArithmeticDataset(
            num_samples=1000,
            phase=3,  # Phase 3: max_depth_val=3에 해당 (3-4자리)
            seed=999,  # 고정된 seed
            mode="val",
            enable_augmentation=False,
        )
        # Validation 샘플을 미리 생성하여 저장
        _FIXED_VAL_SAMPLES = []
        _FIXED_VAL_SAMPLES_INPUTS = set()
        for i in range(len(fixed_val_dataset)):
            sample = fixed_val_dataset[i]
            _FIXED_VAL_SAMPLES.append(sample)
            _FIXED_VAL_SAMPLES_INPUTS.add(sample["input_text"])
    
    # 고정된 validation 샘플을 사용하는 Dataset wrapper
    class FixedValidationDataset:
        def __init__(self, samples):
            self.samples = samples
            self.mode = "val"
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    val_dataset = FixedValidationDataset(_FIXED_VAL_SAMPLES)
    
    # Validation DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        mode="val",
    )

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

    # W&B sweep 최적값 적용
    model_config = ModelConfig(
        d_model=256,  # W&B sweep 최적값
        nhead=2,  # W&B sweep 최적값
        num_encoder_layers=6,  # W&B sweep 최적값
        num_decoder_layers=2,  # W&B sweep 최적값
        dim_feedforward=1024,  # W&B sweep 최적값
        dropout=0.0,  # W&B sweep 최적값 (과적합 없음)
    )

    train_config = TrainConfig(
        max_train_steps=None,
        lr=1e-4,  # EC fine-tuning용 낮춤 (기존 2e-4 → 1e-4)
        warmup_steps=3000,  # Warmup 단축 (fine-tuning이므로)
        weight_decay=0.1,  # 문헌 권장
        grad_clip=1.0,  # 문헌 권장
        valid_every=200,
        max_gen_len=50,
        show_valid_samples=10,  # 카테고리별 10개
        num_epochs=10,  # EC fine-tuning: 10 epochs
        save_best_path="best_model_ec.pt",
        use_cosine_schedule=True,  # cosine decay 유지
        enable_early_stopping=False,  # main()에서는 early stopping 비활성화 (전체 학습)
        early_stopping_patience=5,
        min_lr_threshold=1e-6,
        min_em_threshold=0.01,
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

    model = TransformerSeq2Seq(

    in_vocab=input_tokenizer.vocab_size,

    out_vocab=output_tokenizer.vocab_size,

     **model_config.__dict__,)

    # --------------------------------------------------------------------------
    # 체크포인트 로드 (Resume training)
    # --------------------------------------------------------------------------
    
    resume_checkpoint = "best_model.pt"  # 체크포인트 파일 경로
    resume_from_checkpoint = True  # True로 설정하면 체크포인트에서 재개
    
    if resume_from_checkpoint and os.path.exists(resume_checkpoint):
        print("=" * 70)
        print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
        print("=" * 70)
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            
            # 저장된 설정 확인 및 비교
            if "model_config" in checkpoint:
                saved_config = checkpoint["model_config"]
                print("\n📋 Checkpoint Configuration:")
                config_match = True
                for key in ["d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward", "dropout"]:
                    current_val = model_config.__dict__[key]
                    saved_val = saved_config[key]
                    match_symbol = "✅" if current_val == saved_val else "⚠️"
                    print(f"  {match_symbol} {key:20s}: current={current_val:6}, saved={saved_val:6}")
                    if current_val != saved_val:
                        config_match = False
                
                if not config_match:
                    print("\n⚠️  WARNING: Model configuration mismatch detected!")
                    print("   Using current configuration. Weights may not load correctly.")
                else:
                    print("\n✅ Configuration matches!")
            
            # 모델 weights 로드
            model.load_state_dict(checkpoint["model_state"])
            print("\n✅ Model weights loaded successfully")
            
            # Step 정보 출력
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
        print(f"⚠️  Checkpoint file not found: {resume_checkpoint}")
        print("Starting training from scratch...")
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

    )

    # --------------------------------------------------------------------------

    # 6) 학습 끝나면 모델 저장

    # --------------------------------------------------------------------------

    torch.save(model.state_dict(), "model.pt")

    print("Saved model.pt")

def train_run():
    global _FIXED_VAL_SAMPLES, _FIXED_VAL_SAMPLES_INPUTS
    
    # GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb.init: sweep에서는 config를 넘겨주지 않고, agent가 알아서 주입
    with wandb.init(project="inthon-2025-arithmetic"):
        cfg = wandb.config
        run_name = wandb.run.name  # 미리 저장 (finish 후 접근 불가)

        # ----------------------------------------------------------------------
        # 1) 고정된 Validation 데이터셋 생성 (한 번만)
        # ----------------------------------------------------------------------
        if _FIXED_VAL_SAMPLES is None:
            # 고정된 validation dataset 생성 (W&B sweep 최적값 적용: max_depth_val=3 → phase=3)
            fixed_val_dataset = ArithmeticDataset(
                num_samples=1000,
                phase=3,  # Phase 3: max_depth_val=3에 해당 (3-4자리)
                seed=999,  # 고정된 seed
                mode="val",
                enable_augmentation=False,
            )
            # Validation 샘플을 미리 생성하여 저장
            _FIXED_VAL_SAMPLES = []
            _FIXED_VAL_SAMPLES_INPUTS = set()
            for i in range(len(fixed_val_dataset)):
                sample = fixed_val_dataset[i]
                _FIXED_VAL_SAMPLES.append(sample)
                _FIXED_VAL_SAMPLES_INPUTS.add(sample["input_text"])
        
        # 고정된 validation 샘플로 dataset 생성
        # 고정된 validation 샘플을 사용하는 간단한 Dataset wrapper
        class FixedValidationDataset:
            def __init__(self, samples):
                self.samples = samples
                self.mode = "val"  # mode 속성 추가 (get_dataloader에서 사용)
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        val_dataset = FixedValidationDataset(_FIXED_VAL_SAMPLES)
        val_dataloader = get_dataloader(
            val_dataset,
            batch_size=cfg.batch_size,
            num_workers=0,
            pin_memory=True,
            mode="val",
        )

        # ----------------------------------------------------------------------
        # 2) Train 데이터 준비 (validation 샘플 제외)
        # ----------------------------------------------------------------------
        # Train dataset 생성 시 validation 샘플과 겹치지 않도록 다른 seed 사용
        # (validation seed=999, train seed=123으로 이미 다름)
        # W&B sweep 최적값 적용: max_depth_train=2 → phase=2
        train_dataset = ArithmeticDataset(
            num_samples=200_000,  # 적당한 샘플 수
            phase=cfg.get("phase", 2),  # 기본값: phase 2 (W&B sweep 최적값)
            seed=123,  # Train용 고정 seed (validation과 다름)
            mode="train",
            enable_augmentation=True,
        )

        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=0,
            pin_memory=True,
        )

        # ----------------------------------------------------------------------

        # 2) 토크나이저 설정 및 생성

        # ----------------------------------------------------------------------

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

        # ----------------------------------------------------------------------

        # 3) 모델 설정 (cfg 기반)

        # ----------------------------------------------------------------------

        model_config = ModelConfig(

            d_model=cfg.d_model,

            nhead=cfg.nhead,  # ⚠️ model.py에서 n_head 또는 nhead 잘 맞춰줘야 함

            num_encoder_layers=cfg.num_encoder_layers,

            num_decoder_layers=cfg.num_decoder_layers,

            dim_feedforward=cfg.dim_feedforward,

            dropout=cfg.dropout,

        )

        # ----------------------------------------------------------------------

        # 4) 학습 설정 (cfg 기반)

        # ----------------------------------------------------------------------

        # TrainConfig 생성 시 cfg에서 필요한 값만 명시적으로 전달
        # (Wandb가 cfg에 예상치 못한 키를 추가할 수 있으므로 명시적으로 처리)
        train_config = TrainConfig(
            max_train_steps=None,
            lr=cfg.lr,
            warmup_steps=5000,  # 고정값: warmup 5k steps (문헌 권장)
            weight_decay=0.1,  # 고정값: weight decay 0.1 (문헌 권장)
            grad_clip=1.0,  # 고정값: grad clip 1.0 (문헌 권장)
            valid_every=200,
            max_gen_len=50,
            show_valid_samples=10,  # 카테고리별 10개
            num_epochs=20,  # W&B sweep 최적값: 20 epochs
            save_best_path=f"best_{run_name}.pt",  # run_name 미리 저장한 값 사용
            use_cosine_schedule=True,  # cosine decay 유지
            enable_early_stopping=True,  # wandb sweep용 early stopping 활성화
            early_stopping_patience=5,  # 5번의 validation 동안 개선 없으면 종료
            min_lr_threshold=1e-6,  # 학습률이 1e-6 이하로 떨어지면 종료
            min_em_threshold=0.01,  # EM이 0.01 이하이고 patience 초과 시 종료
        )

        # ----------------------------------------------------------------------

        # 5) 모델 생성

        # ----------------------------------------------------------------------

        model = TransformerSeq2Seq(

            in_vocab=input_tokenizer.vocab_size,

            out_vocab=output_tokenizer.vocab_size,

            **model_config.__dict__,  # d_model, n_head, num_layers 등 전달

        )

        # ----------------------------------------------------------------------

        # 6) 학습 시작

        # ----------------------------------------------------------------------

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

        )

        # 원하면 각 run 끝에 최종 모델도 따로 저장 가능
        # wandb.run.name은 finish 후 접근 불가하므로 미리 저장한 값 사용
        torch.save(model.state_dict(), "model_last.pt")
        print("Saved model_last.pt for run:", run_name)

# python train.py로 실행했을 때만 main()을 돌게 합니다.

if __name__ == "__main__":

    main()

    #sweep_id = wandb.sweep(sweep_config, project="inthon-2025-arithmetic")

    #wandb.agent(sweep_id, function=train_run, count=20)  # 20번 실험 (원하는 만큼 조정)
