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
    """모델 아키텍처 관련 설정 (legacy best baseline)"""
    d_model: int = 256  # Hidden dimension (legacy 최적값)
    nhead: int = 2  # Attention heads (legacy 최적값)
    num_encoder_layers: int = 6  # Encoder layers (legacy 최적값)
    num_decoder_layers: int = 2  # Decoder layers (legacy 최적값)
    dim_feedforward: int = 1024  # FFN dimension (legacy 최적값)
    dropout: float = 0.0  # Dropout (legacy 최적값: 0으로 과적합 없음)
    # 기존 best_model.pt와 완전 호환되는 기본 아키텍처

DEPTH_PROFILES = {
    "baseline": {
        "num_encoder_layers": 6,
        "num_decoder_layers": 2,
    },
    "legacy_powerup": {
        "num_encoder_layers": 8,
        "num_decoder_layers": 3,
    },
    "deep_context": {
        "num_encoder_layers": 10,
        "num_decoder_layers": 4,
    },
    "ultra_context": {
        "num_encoder_layers": 12,
        "num_decoder_layers": 4,
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
    """학습 관련 설정 (legacy best baseline)"""
    max_train_steps: Optional[int] = None
    lr: float = 5e-4  # Learning rate (legacy best)
    warmup_steps: int = 5000  # Warmup steps (legacy 설정)
    weight_decay: float = 0.1  # Weight decay (문헌 권장)
    grad_clip: float = 1.0  # Gradient clipping (문헌 권장)
    valid_every: int = 200  # Validation frequency (legacy 설정)
    max_gen_len: int = 50  # Max generation length (문헌 권장: 50)
    show_valid_samples: int = 5  # legacy default
    num_epochs: int = 20  # legacy (W&B 최적)
    batch_size: int = 128  # legacy baseline batch size
    save_best_path: Optional[str] = None
    use_cosine_schedule: bool = True  # Use cosine decay after warmup
    # Early stopping for wandb sweep
    early_stopping_patience: int = 5  # legacy patience
    min_lr_threshold: float = 1e-6  # Minimum learning rate threshold (below this, stop training)
    min_em_threshold: float = 0.01  # Minimum EM threshold (below this after patience, stop)
    enable_early_stopping: bool = True  # Enable early stopping for wandb sweep
    depth_profile: str = "baseline"  # Depth preset identifier (baseline, legacy_powerup, ...)
    train_num_samples: int = 900_000  # Large-scale training set (>= 50만)
    val_num_samples: int = 3_000  # Larger validation coverage
    train_phase_mix: Tuple[int, ...] = (2, 3, 4)  # Phase mixture for training diversity
    val_phase: int = 4  # Validation focuses on hardest distribution
    lambda_rpn: float = 0.0  # Auxiliary loss weight for RPN decoder (0.0 = disabled)


