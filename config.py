from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    d_model: int = 256  # Hidden dimension 크기 (임베딩, GRU hidden size)
    nhead : int = 2
    num_encoder_layers : int = 6
    num_decoder_layers : int = 2
    dim_feedforward : int = 1024
    dropout : float = 0
    # 향후 확장 가능: num_layers, dropout, attention heads 등


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3
    warmup_steps: int = 5000  # Warmup steps 추가 (train.py에서 사용)
    weight_decay: float = 0.1  # Weight decay 추가 (train.py에서 사용)
    grad_clip: float = 1.0  # Gradient clipping 추가 (train.py에서 사용)
    valid_every: int = 50
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 4
    save_best_path: Optional[str] = None
    use_cosine_schedule: bool = True  # Cosine schedule 추가 (train.py에서 사용)
    # Early stopping for wandb sweep
    early_stopping_patience: int = 5
    min_lr_threshold: float = 1e-6
    min_em_threshold: float = 0.01
    enable_early_stopping: bool = True


