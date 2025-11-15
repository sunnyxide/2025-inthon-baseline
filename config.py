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
    """모델 아키텍처 관련 설정 (리뷰 기반 최적화)"""
    d_model: int = 384  # Hidden dimension (문헌 검증: 384는 산술 학습에 최적)
    nhead: int = 6  # Attention heads (d_model 384 → head dim 64)
    num_encoder_layers: int = 4  # Encoder layers
    num_decoder_layers: int = 4  # Decoder layers
    dim_feedforward: int = 1536  # FFN dimension (4×d_model, 표준값)
    dropout: float = 0.1  # Dropout (과적합 시 0.2로 증가)
    # 향후 확장 가능: positional encoding type (NoPE/FIRE), abacus embedding 등


@dataclass
class TrainConfig:
    """학습 관련 설정 (리뷰 기반 최적화)"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3  # Learning rate (AdamW + warmup 5k + cosine decay)
    warmup_steps: int = 5000  # Warmup steps (문헌 권장)
    weight_decay: float = 0.1  # Weight decay (문헌 권장)
    grad_clip: float = 1.0  # Gradient clipping (문헌 권장)
    valid_every: int = 200  # Validation frequency
    max_gen_len: int = 50  # Max generation length (문헌 권장: 50)
    show_valid_samples: int = 5
    num_epochs: int = 10
    save_best_path: Optional[str] = None
    use_cosine_schedule: bool = True  # Use cosine decay after warmup


