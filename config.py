from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정 (RPN 유효성 검증용 - baseline 호환)"""
    d_model: int = 256  # Hidden dimension (checkpoint 호환 유지)
    nhead: int = 2  # Attention heads (checkpoint 호환 - nhead 변경 시 weight 재사용 불가)
    num_encoder_layers: int = 6  # Encoder layers (baseline: best_model.pt와 완전 호환)
    num_decoder_layers: int = 2  # Decoder layers (baseline: best_model.pt와 완전 호환)
    dim_feedforward: int = 1024  # FFN dimension (유지)
    dropout: float = 0.0  # Dropout (유지: 과적합 없음)
    use_digit_conv: bool = False  # Optional 1D conv layer for digit-wise carry modeling
    use_right_aligned_pos: bool = True  # Developer log: 오른쪽 정렬 위치 인코딩으로 digit-wise 알고리즘 강화
    use_token_type_emb: bool = True  # Developer log: 토큰 타입 임베딩으로 구조적 계산 inductive bias 제공
    use_operator_head: bool = True  # Developer log: 연산자별 출력 헤드로 operator-specific circuits 구현
    # 독립변수 테스트: RPN 유효성 검증을 위해 레이어는 baseline 유지

DEPTH_PROFILES = {
    "baseline": {
        "num_encoder_layers": 6,
        "num_decoder_layers": 2,
        "nhead": 2,
    },
    "legacy_powerup": {
        "num_encoder_layers": 8,
        "num_decoder_layers": 3,
        "nhead": 4,
    },
    "deep_context": {
        "num_encoder_layers": 10,
        "num_decoder_layers": 4,
        "nhead": 4,
    },
    "ultra_context": {
        "num_encoder_layers": 12,
        "num_decoder_layers": 4,
        "nhead": 4,
    },
}


def apply_depth_profile(base_config: ModelConfig, profile: str | None) -> ModelConfig:
    """
    Legacy best 하이퍼파라미터(d_model/nhead/FFN)는 유지한 채 encoder/decoder depth만 조정.
    """
    profile_key = (profile or "baseline").lower()
    overrides = DEPTH_PROFILES.get(profile_key)
    if overrides is None:
        available = ", ".join(sorted(DEPTH_PROFILES.keys()))
        raise ValueError(f"Unknown depth profile '{profile_key}'. Available: {available}")

    cfg_dict = asdict(base_config)
    cfg_dict.update(overrides)
    return ModelConfig(**cfg_dict)


@dataclass
class TrainConfig:
    """학습 관련 설정 (RPN 유효성 검증용 - 독립변수 테스트)"""
    max_train_steps: Optional[int] = 120_000  # Global max steps (cosine schedule 기준 상한)
    lr: float = 2e-4  # Learning rate (5e-4 → 2e-4: 안정적 수렴용 기본 lr)
    warmup_steps: int = 7000  # Warmup steps (5000 → 7000: 긴 학습에 맞춘 완만한 워밍업)
    weight_decay: float = 0.1  # Weight decay (문헌 권장)
    grad_clip: float = 1.0  # Gradient clipping (문헌 권장)
    valid_every: int = 200  # Validation frequency (legacy 설정)
    max_gen_len: int = 50  # Max generation length (문헌 권장: 50)
    show_valid_samples: int = 10  # 5 → 10 (다양한 검증 샘플 확인)
    num_epochs: int = 15  # 20 → 15 (학습 시간 단축)
    batch_size: int = 128  # legacy baseline batch size
    save_best_path: Optional[str] = None
    use_cosine_schedule: bool = True  # Use cosine decay after warmup
    # Early stopping for wandb sweep
    early_stopping_patience: int = 8  # 5 → 8 (모델 용량 증가로 patience 증가)
    min_lr_threshold: float = 1e-5  # Minimum learning rate threshold (below this, stop training)
    min_em_threshold: float = 0.01  # Minimum EM threshold (below this after patience, stop)
    enable_early_stopping: bool = True  # Enable early stopping for wandb sweep
    depth_profile: str = "baseline"  # Depth preset identifier (baseline, legacy_powerup, ...)
    train_num_samples: int = 600_000  # EC enhanced dataset (40만 base + 20만 augmentation)
    val_num_samples: int = 3_000  # Larger validation coverage
    train_phase_mix: Tuple[int, ...] = (2, 3, 4)  # Phase mixture for training diversity
    val_phase: int = 4  # Validation focuses on hardest distribution
    lambda_rpn: float = 0.2  # Auxiliary loss weight for RPN decoder (독립변수: RPN 유효성 검증)
    lambda_ec: float = 0.1  # Expression consistency loss weight for EC 동치 수식 그룹
    lambda_rpn_value: float = 0.0  # RPN scratchpad stack-value regression loss weight
    # Developer log: Optional validation-metric 기반 Plateau 스케줄러 설정
    use_plateau_schedule: bool = False  # True일 때 ReduceLROnPlateau 사용
    plateau_factor: float = 0.5  # Plateau에서 lr 감소 비율
    plateau_patience: int = 2  # Plateau 스케줄러 patience (validation 기준)


