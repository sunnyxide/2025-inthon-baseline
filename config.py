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
    """모델 아키텍처 관련 설정 (A100 GPU 최적화)"""
    d_model: int = 512  # Hidden dimension (A100 최적화: 256 → 512)
    nhead: int = 8  # Attention heads (A100 최적화: 2 → 8)
    num_encoder_layers: int = 8  # Encoder layers (A100 최적화: 6 → 8)
    num_decoder_layers: int = 4  # Decoder layers (A100 최적화: 2 → 4)
    dim_feedforward: int = 2048  # FFN dimension (A100 최적화: 1024 → 2048)
    dropout: float = 0.0  # Dropout (W&B sweep 최적값: 0으로 과적합 방지 불필요)
    # A100 GPU에 최적화된 대형 모델 설정
    # 이전 best_model.pt(d_model=256)와 호환되지 않음 - 새로운 학습 시작 필요


@dataclass
class TrainConfig:
    """학습 관련 설정 (A100 GPU 최적화)"""
    max_train_steps: Optional[int] = None
    lr: float = 3e-4  # Learning rate (A100용 대형 모델: 5e-4 → 3e-4로 안정화)
    warmup_steps: int = 8000  # Warmup steps (대형 데이터셋용: 5000 → 8000)
    weight_decay: float = 0.1  # Weight decay (문헌 권장)
    grad_clip: float = 1.0  # Gradient clipping (문헌 권장)
    valid_every: int = 500  # Validation frequency (대형 데이터셋: 200 → 500)
    max_gen_len: int = 50  # Max generation length (문헌 권장: 50)
    show_valid_samples: int = 10  # Sample display count (5 → 10)
    num_epochs: int = 30  # Epochs (대형 데이터셋: 20 → 30)
    batch_size: int = 512  # Batch size (A100 최적화: 128 → 512)
    save_best_path: Optional[str] = None
    use_cosine_schedule: bool = True  # Use cosine decay after warmup
    # Early stopping for wandb sweep
    early_stopping_patience: int = 8  # Patience (대형 모델: 5 → 8)
    min_lr_threshold: float = 1e-6  # Minimum learning rate threshold (below this, stop training)
    min_em_threshold: float = 0.01  # Minimum EM threshold (below this after patience, stop)
    enable_early_stopping: bool = True  # Enable early stopping for wandb sweep


