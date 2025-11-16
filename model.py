from __future__ import annotations

from typing import List, Dict, Any, Tuple

from dataclasses import dataclass

import torch

import torch.nn as nn

import math

from do_not_edit.model_template import BaseModel

# ========================
# Tokenizer (통합 버전)
# ========================

# 특수 토큰 정의
# PAD: 패딩 토큰 (배치 처리 시 길이를 맞추기 위해 사용)
# BOS: Beginning of Sequence (시퀀스 시작 토큰)
# EOS: End of Sequence (시퀀스 종료 토큰)
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

# 규정에 맞는 입력/출력 문자 집합
# INPUT_CHARS: 수식 입력에 사용 가능한 모든 문자 (숫자, 연산자, 괄호)
# OUTPUT_CHARS: 모델이 출력할 수 있는 문자 (숫자만)
INPUT_CHARS = list("0123456789+-*/()")
OUTPUT_CHARS = list("0123456789")


class CharTokenizer:
    """
    문자 단위 토크나이저
    
    문자열을 문자 단위로 분해하여 정수 인덱스로 변환하는 토크나이저입니다.
    Seq2Seq 모델의 입력/출력을 처리하기 위해 사용됩니다.
    
    Attributes:
        stoi (Dict[str, int]): 문자 → 인덱스 매핑 딕셔너리
        itos (Dict[int, str]): 인덱스 → 문자 매핑 딕셔너리
        pad (str | None): 패딩 토큰 문자열 (add_special=True일 때만 설정)
        bos (str | None): 시작 토큰 문자열 (add_special=True일 때만 설정)
        eos (str | None): 종료 토큰 문자열 (add_special=True일 때만 설정)
    """
    
    def __init__(self, chars: List[str], add_special: bool):
        """
        토크나이저 초기화
        
        Args:
            chars: 토크나이저에 포함할 문자 리스트
            add_special: True일 경우 PAD, BOS, EOS 특수 토큰을 vocab에 추가
        """
        # 기본 vocab은 입력받은 문자 리스트
        vocab = list(chars)
        
        # 특수 토큰 설정 (add_special이 True일 때만 사용)
        self.pad = PAD if add_special else None
        self.bos = BOS if add_special else None
        self.eos = EOS if add_special else None
        
        # 특수 토큰을 vocab 앞에 추가 (순서: PAD, BOS, EOS, ...chars)
        if add_special:
            vocab = [PAD, BOS, EOS] + vocab
        
        # 문자 → 인덱스 매핑 생성
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        # 인덱스 → 문자 매핑 생성 (역변환용)
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str, add_bos_eos: bool) -> List[int]:
        """
        문자열을 정수 인덱스 리스트로 변환 (인코딩)
        
        Args:
            s: 인코딩할 문자열
            add_bos_eos: True일 경우 BOS와 EOS 토큰을 앞뒤에 추가
            
        Returns:
            정수 인덱스 리스트
            
        Raises:
            ValueError: vocab에 없는 문자가 포함된 경우
        """
        ids: List[int] = []
        
        # BOS 토큰 추가 (시퀀스 시작 표시)
        if add_bos_eos and self.bos is not None:
            ids.append(self.stoi[self.bos])
        
        # 각 문자를 인덱스로 변환
        for ch in s:
            idx = self.stoi.get(ch, None)
            if idx is None:
                # vocab에 없는 문자는 오류 발생 (조용한 실패 방지)
                raise ValueError(f"Unknown char '{ch}' for tokenizer.")
            ids.append(idx)
        
        # EOS 토큰 추가 (시퀀스 종료 표시)
        if add_bos_eos and self.eos is not None:
            ids.append(self.stoi[self.eos])
        
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """
        정수 인덱스 리스트를 문자열로 변환 (디코딩)
        
        Args:
            ids: 디코딩할 정수 인덱스 리스트
            strip_special: True일 경우 특수 토큰(PAD, BOS, EOS)을 제거
            
        Returns:
            디코딩된 문자열
        """
        # 인덱스를 문자로 변환하여 문자열 생성
        s = "".join(self.itos[i] for i in ids if i in self.itos)
        
        # 특수 토큰 제거 (strip_special=True일 때)
        if strip_special and self.bos:
            s = s.replace(self.bos, "")
        if strip_special and self.eos:
            s = s.replace(self.eos, "")
        if strip_special and self.pad:
            s = s.replace(self.pad, "")
        
        return s

    @property
    def vocab_size(self) -> int:
        """vocab 크기 반환"""
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        """패딩 토큰의 인덱스 반환 (없으면 0)"""
        return self.stoi.get(PAD, 0)

    @property
    def bos_id(self) -> int:
        """시작 토큰의 인덱스 반환 (없으면 0)"""
        return self.stoi.get(BOS, 0)

    @property
    def eos_id(self) -> int:
        """종료 토큰의 인덱스 반환 (없으면 0)"""
        return self.stoi.get(EOS, 0)


@dataclass
class BatchTensors:
    """
    배치 처리용 텐서 컨테이너
    
    Attributes:
        src: 소스(입력) 시퀀스 텐서 [batch_size, src_len]
        tgt_inp: 타겟(출력) 입력 시퀀스 텐서 [batch_size, tgt_len] (teacher forcing용)
        tgt_out: 타겟(출력) 정답 시퀀스 텐서 [batch_size, tgt_len] (loss 계산용)
    """
    src: torch.Tensor
    tgt_inp: torch.Tensor
    tgt_out: torch.Tensor


def _pad(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    """
    시퀀스 리스트를 패딩하여 동일한 길이의 텐서로 변환
    
    배치 내 시퀀스들의 길이가 다를 때, 가장 긴 시퀀스 길이에 맞춰
    짧은 시퀀스는 pad_id로 패딩합니다.
    
    Args:
        seqs: 정수 인덱스 리스트의 리스트 (각 리스트가 하나의 시퀀스)
        pad_id: 패딩에 사용할 토큰 인덱스
        
    Returns:
        패딩된 텐서 [batch_size, max_len]
    """
    # 가장 긴 시퀀스 길이 계산
    L = max(len(s) for s in seqs) if len(seqs) > 0 else 1
    
    # 패딩으로 채워진 텐서 생성 [batch_size, max_len]
    out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    
    # 각 시퀀스를 텐서에 복사
    for i, s in enumerate(seqs):
        if len(s) > 0:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    
    return out


def tokenize_batch(
    batch: Dict[str, List[str]],
    input_tokenizer: CharTokenizer,
    output_tokenizer: CharTokenizer,
) -> BatchTensors:
    """
    배치 데이터를 토크나이징하여 모델 입력용 텐서로 변환
    
    학습 시 teacher forcing을 위해 tgt_inp와 tgt_out을 별도로 생성합니다.
    - tgt_inp: 디코더 입력 (BOS + target_text, EOS 제외)
    - tgt_out: 디코더 출력 정답 (target_text + EOS)
    
    Args:
        batch: {"input_text": [...], "target_text": [...]} 형식의 딕셔너리
        input_tokenizer: 입력 시퀀스 토크나이저
        output_tokenizer: 출력 시퀀스 토크나이저
        
    Returns:
        BatchTensors 객체 (src, tgt_inp, tgt_out 텐서 포함)
        
    Raises:
        ValueError: 토크나이징 결과가 빈 시퀀스인 경우
    """
    # 입력 시퀀스 토크나이징 (BOS/EOS 없이)
    src_ids = [input_tokenizer.encode(s, add_bos_eos=False) for s in batch["input_text"]]
    
    # 빈 시퀀스 검사 (조용한 실패 방지)
    empty_indices = [i for i, seq in enumerate(src_ids) if len(seq) == 0]
    if empty_indices:
        inputs = [batch["input_text"][i] for i in empty_indices]
        raise ValueError(f"Empty tokenized source at indices {empty_indices}; inputs: {inputs}")
    
    # 타겟 입력 시퀀스 생성 (teacher forcing용)
    # 형식: BOS + target_text (EOS는 제외, 디코더가 예측해야 함)
    tgt_inp_ids = [
        output_tokenizer.encode("", add_bos_eos=True)[:-1] +  # BOS만 가져오기 ([:-1]로 EOS 제거)
        output_tokenizer.encode(batch["target_text"][i], add_bos_eos=False)
        for i in range(len(batch["target_text"]))
    ]
    
    # 타겟 출력 시퀀스 생성 (loss 계산용)
    # 형식: target_text + EOS
    tgt_out_ids = [
        output_tokenizer.encode(batch["target_text"][i], add_bos_eos=False) +
        [output_tokenizer.eos_id]  # EOS 추가
        for i in range(len(batch["target_text"]))
    ]
    
    # 패딩하여 텐서로 변환
    src = _pad(src_ids, input_tokenizer.pad_id)
    tgt_inp = _pad(tgt_inp_ids, output_tokenizer.pad_id)
    tgt_out = _pad(tgt_out_ids, output_tokenizer.pad_id)
    
    return BatchTensors(src=src, tgt_inp=tgt_inp, tgt_out=tgt_out)


# ========================
# TinySeq2Seq (baseline)
# ========================

class TinySeq2Seq(nn.Module):
    """
    매우 단순한 GRU 기반 Seq2Seq 모델
    
    구조:
    - 인코더: GRU를 사용하여 입력 시퀀스를 고정 크기 hidden state로 변환
    - 디코더: GRU를 사용하여 hidden state로부터 출력 시퀀스를 생성
    - 생성: Greedy decoding 방식 사용
    
    API:
      forward(src, tgt_inp, src_pad_id) -> logits [B, T, V]
      generate(src, max_len, bos_id, eos_id, src_pad_id) -> ids [B, T']
    """
    
    def __init__(self, in_vocab: int, out_vocab: int, **kwargs):
        """
        모델 초기화
        
        Args:
            in_vocab: 입력 vocab 크기
            out_vocab: 출력 vocab 크기
            **kwargs: 모델 설정 (d_model 등)
        """
        super().__init__()
        
        # 모델 설정 추출 (기본값 포함)
        d_model = kwargs.get("d_model", 256)
        
        # 입력 임베딩 레이어 (입력 문자 → d_model 차원 벡터)
        self.embed_in = nn.Embedding(in_vocab, d_model)
        
        # 인코더 GRU (입력 시퀀스를 hidden state로 변환)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        
        # 출력 임베딩 레이어 (출력 문자 → d_model 차원 벡터)
        self.embed_out = nn.Embedding(out_vocab, d_model)
        
        # 디코더 GRU (hidden state로부터 출력 시퀀스 생성)
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        
        # 출력 프로젝션 레이어 (hidden state → vocab 크기 logits)
        self.out_proj = nn.Linear(d_model, out_vocab)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor, src_pad_id: int, teacher_forcing: float = 1.0) -> torch.Tensor:
        """
        순전파 (학습 시 사용)
        
        Args:
            src: 입력 시퀀스 텐서 [batch_size, src_len]
            tgt_inp: 타겟 입력 시퀀스 텐서 [batch_size, tgt_len] (teacher forcing용)
            src_pad_id: 소스 패딩 토큰 ID (현재 구현에서는 사용하지 않음)
            teacher_forcing: teacher forcing 비율 (현재 구현에서는 항상 1.0)
            
        Returns:
            각 타임스텝의 vocab 크기 logits [batch_size, tgt_len, out_vocab]
        """
        # Encoder: 입력 시퀀스를 hidden state로 변환
        x = self.embed_in(src)  # [B, src_len, d_model]
        enc_out, h = self.encoder(x)  # h: [1, B, d_model] (마지막 hidden state)
        
        # Decoder: teacher forcing을 통해 전체 타겟 시퀀스를 한 번에 처리
        y = self.embed_out(tgt_inp)  # [B, tgt_len, d_model]
        dec_out, _ = self.decoder(y, h)  # [B, tgt_len, d_model]
        
        # 각 타임스텝의 vocab 크기 logits 생성
        logits = self.out_proj(dec_out)  # [B, tgt_len, out_vocab]
        
        return logits

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int, src_pad_id: int) -> torch.Tensor:
        """
        추론 시 시퀀스 생성 (Greedy Decoding)
        
        BOS 토큰부터 시작하여 EOS 토큰이 나올 때까지 또는 max_len에 도달할 때까지
        각 타임스텝마다 가장 확률이 높은 토큰을 선택합니다.
        
        Args:
            src: 입력 시퀀스 텐서 [batch_size, src_len]
            max_len: 생성할 최대 시퀀스 길이
            bos_id: 시작 토큰 ID
            eos_id: 종료 토큰 ID
            src_pad_id: 소스 패딩 토큰 ID (현재 구현에서는 사용하지 않음)
            
        Returns:
            생성된 토큰 ID 시퀀스 [batch_size, gen_len]
        """
        # Encoder: 입력 시퀀스를 hidden state로 변환
        x = self.embed_in(src)  # [B, src_len, d_model]
        enc_out, h = self.encoder(x)  # h: [1, B, d_model]
        
        B = src.size(0)  # 배치 크기
        
        # BOS 토큰으로 시작
        y = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        
        outputs = []  # 생성된 토큰들을 저장할 리스트
        
        # 최대 max_len까지 토큰 생성
        for _ in range(max_len):
            # 마지막 토큰만 사용하여 다음 토큰 예측
            emb = self.embed_out(y[:, -1:])  # [B, 1, d_model]
            dec_out, h = self.decoder(emb, h)  # [B, 1, d_model]
            
            # vocab 크기 logits 생성
            logits = self.out_proj(dec_out[:, -1, :])  # [B, out_vocab]
            
            # Greedy decoding: 가장 확률이 높은 토큰 선택
            next_id = torch.argmax(logits, dim=-1)  # [B]
            
            outputs.append(next_id)
            
            # 생성된 토큰을 시퀀스에 추가
            y = torch.cat([y, next_id.unsqueeze(1)], dim=1)
            
            # 모든 배치에서 EOS가 생성되면 중단
            if torch.all(next_id == eos_id):
                break
        
        # 생성된 토큰들을 텐서로 변환
        if outputs:
            return torch.stack(outputs, dim=1)  # [B, gen_len]
        
        # 생성된 토큰이 없으면 빈 텐서 반환
        return torch.empty((B, 0), dtype=torch.long, device=src.device)


class PositionalEncoding(nn.Module):
    """
    표준 sin/cos 위치 인코딩.
    입력: [B, T, d_model]
    출력: [B, T, d_model] (positional encoding이 더해진 결과)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [T, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 짝수 차원
        pe[:, 1::2] = torch.cos(position * div_term)   # 홀수 차원
        pe = pe.unsqueeze(0)  # [1, T, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    디코더용 causal mask (각 위치가 이전 토큰까지만 볼 수 있도록).
    반환: [T, T] float 텐서 (masked 위치는 -inf, 나머지는 0)
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)  # 상삼각(대각선 위) = 1
    mask = mask.masked_fill(mask == 1, float("-inf"))  # 가려야 할 곳은 -inf
    mask = mask.masked_fill(mask == 0, 0.0)            # 나머지는 0
    return mask


class TransformerSeq2Seq(nn.Module):
    """
    문자 단위 Transformer 기반 Seq2Seq 모델.
    - 인코더: TransformerEncoder
    - 디코더: TransformerDecoder
    - forward:
        src  : [B, S]
        tgt_inp : [B, T] (BOS + target tokens)
        -> logits: [B, T, out_vocab]
    - generate:
        src  : [B, S]
        -> greedy decoding으로 [B, T'] 생성
    """
    def __init__(self, in_vocab: int, out_vocab: int, **kwargs):
        super().__init__()
        # 기본 하이퍼파라미터 설정 (model_config_dict에서 override 가능)
        d_model = kwargs.get("d_model", 256)
        nhead = kwargs.get("nhead", 4)
        num_encoder_layers = kwargs.get("num_encoder_layers", 4)
        num_decoder_layers = kwargs.get("num_decoder_layers", 4)
        dim_feedforward = kwargs.get("dim_feedforward", 512)
        dropout = kwargs.get("dropout", 0.1)
        rpn_vocab = kwargs.get("rpn_vocab")
        use_digit_conv = kwargs.get("use_digit_conv", False)
        self.d_model = d_model
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.rpn_vocab = rpn_vocab
        self.use_digit_conv = use_digit_conv

        # 임베딩
        self.embed_in = nn.Embedding(in_vocab, d_model)
        self.embed_out = nn.Embedding(out_vocab, d_model)

        # 위치 인코딩
        self.pos_enc_in = PositionalEncoding(d_model, dropout)
        self.pos_enc_out = PositionalEncoding(d_model, dropout)

        # 인코더 / 디코더 레이어 정의 (batch_first=True로 [B, T, C] 사용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 출력 projection
        self.out_proj = nn.Linear(d_model, out_vocab)

        # Digit-level 1D convolution module (선택적 자리올림 패턴 보강용)
        if self.use_digit_conv:
            # Developer log: 1D conv over decoder time dimension for carry modeling
            self.digit_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            self.digit_conv_activation = nn.ReLU()
        else:
            self.digit_conv = None
            self.digit_conv_activation = None

        # RPN auxiliary decoder/projection (훈련 전용, optional)
        if rpn_vocab is not None:
            rpn_decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.rpn_decoder = nn.TransformerDecoder(rpn_decoder_layer, num_layers=num_decoder_layers)
            self.rpn_embed = nn.Embedding(rpn_vocab, d_model)
            self.rpn_out = nn.Linear(d_model, rpn_vocab)
            # Developer log: RPN stack-value regression head (scratchpad supervision)
            self.rpn_value_out = nn.Linear(d_model, 1)
        else:
            self.rpn_decoder = None
            self.rpn_embed = None
            self.rpn_out = None
            self.rpn_value_out = None

    def forward(
        self,
        src: torch.Tensor,       # [B, S]
        tgt_inp: torch.Tensor,   # [B, T]
        src_pad_id: int,
        teacher_forcing: float = 1.0,  # 현재는 항상 teacher forcing 1.0 (GRU 버전과 동일)
    ) -> torch.Tensor:
        """
        학습 시 순전파.
        반환: logits [B, T, out_vocab]
        """
        device = src.device
        B, S = src.size()
        _, T = tgt_inp.size()

        # --- Encoder ---
        src_emb = self.embed_in(src) * math.sqrt(self.d_model)  # [B, S, d_model]
        src_emb = self.pos_enc_in(src_emb)                      # 위치 인코딩 추가

        # 소스 패딩 마스크 (True = 패딩임)
        src_key_padding_mask = (src == src_pad_id)  # [B, S], bool

        # 인코더 출력 (memory)
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, S, d_model]

        # --- Decoder ---
        tgt_emb = self.embed_out(tgt_inp) * math.sqrt(self.d_model)  # [B, T, d_model]
        tgt_emb = self.pos_enc_out(tgt_emb)

        # causal mask: 디코더가 미래 토큰을 못 보도록
        tgt_mask = _generate_square_subsequent_mask(T, device=device)  # [T, T]

        # (단순화 위해 tgt_key_padding_mask는 사용하지 않음: GRU 버전과 일관되게)
        dec_out = self.decoder(
            tgt_emb,           # [B, T, d_model]
            memory,            # [B, S, d_model]
            tgt_mask=tgt_mask, # [T, T]
            memory_key_padding_mask=src_key_padding_mask,
        )  # [B, T, d_model]

        # 선택적 digit-level 1D conv로 자리올림/자리수 상호작용 보강
        if self.use_digit_conv and self.digit_conv is not None:
            conv_in = dec_out.transpose(1, 2)  # [B, C, T]
            conv_out = self.digit_conv(conv_in)
            conv_out = self.digit_conv_activation(conv_out)
            dec_out = conv_out.transpose(1, 2)  # [B, T, C]

        logits = self.out_proj(dec_out)  # [B, T, out_vocab]
        return logits

    def forward_with_rpn(
        self,
        src: torch.Tensor,
        tgt_result_inp: torch.Tensor,
        src_pad_id: int,
        rpn_inp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        학습 시 결과/보조(RPN) 디코더를 동시에 실행.
        Returns:
            result_logits: [B, T_res, out_vocab]
            rpn_logits   : [B, T_rpn, out_vocab]
        """
        if self.rpn_decoder is None or self.rpn_embed is None or self.rpn_out is None:
            raise RuntimeError("RPN head is not initialized; pass rpn_vocab when constructing the model.")
        device = src.device
        B, S = src.size()
        _, T_result = tgt_result_inp.size()
        _, T_rpn = rpn_inp.size()

        # --- Encoder ---
        src_emb = self.embed_in(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc_in(src_emb)
        src_key_padding_mask = (src == src_pad_id)
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        # --- Result decoder ---
        res_emb = self.embed_out(tgt_result_inp) * math.sqrt(self.d_model)
        res_emb = self.pos_enc_out(res_emb)
        res_mask = _generate_square_subsequent_mask(T_result, device=device)
        res_out = self.decoder(
            res_emb,
            memory,
            tgt_mask=res_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        if self.use_digit_conv and self.digit_conv is not None:
            conv_in_res = res_out.transpose(1, 2)
            conv_out_res = self.digit_conv(conv_in_res)
            conv_out_res = self.digit_conv_activation(conv_out_res)
            res_out = conv_out_res.transpose(1, 2)
        result_logits = self.out_proj(res_out)

        # --- RPN decoder (훈련 전용) ---
        rpn_emb = self.rpn_embed(rpn_inp) * math.sqrt(self.d_model)
        rpn_emb = self.pos_enc_out(rpn_emb)
        rpn_mask = _generate_square_subsequent_mask(T_rpn, device=device)
        rpn_out = self.rpn_decoder(
            rpn_emb,
            memory,
            tgt_mask=rpn_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        rpn_logits = self.rpn_out(rpn_out)
        rpn_value = None
        if self.rpn_value_out is not None:
            # [B, T_rpn]
            rpn_value = self.rpn_value_out(rpn_out).squeeze(-1)

        return result_logits, rpn_logits, rpn_value

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,   # [B, S]
        max_len: int,
        bos_id: int,
        eos_id: int,
        src_pad_id: int,
    ) -> torch.Tensor:
        """
        추론 시 greedy decoding.
        반환: [B, T'] (EOS 나오면 중단, 아니면 max_len까지)
        """
        device = src.device
        B, S = src.size()

        # --- Encoder ---
        src_emb = self.embed_in(src) * math.sqrt(self.d_model)    # [B, S, d_model]
        src_emb = self.pos_enc_in(src_emb)
        src_key_padding_mask = (src == src_pad_id)                # [B, S]
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, S, d_model]

        # 디코더 입력 시작: BOS
        y = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # [B, 1]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs = []

        for _ in range(max_len):
            T = y.size(1)
            tgt_emb = self.embed_out(y) * math.sqrt(self.d_model)  # [B, T, d_model]
            tgt_emb = self.pos_enc_out(tgt_emb)
            tgt_mask = _generate_square_subsequent_mask(T, device=device)  # [T, T]

            dec_out = self.decoder(
                tgt_emb,       # [B, T, d_model]
                memory,        # [B, S, d_model]
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )  # [B, T, d_model]

            # Digit-level conv (optional)
            if self.use_digit_conv and self.digit_conv is not None:
                conv_in = dec_out.transpose(1, 2)  # [B, C, T]
                conv_out = self.digit_conv(conv_in)
                conv_out = self.digit_conv_activation(conv_out)
                dec_out = conv_out.transpose(1, 2)

            last_step = dec_out[:, -1, :]        # [B, d_model]
            logits = self.out_proj(last_step)    # [B, out_vocab]
            next_id = torch.argmax(logits, dim=-1)  # [B]
            outputs.append(next_id)

            # y에 토큰 추가
            y = torch.cat([y, next_id.unsqueeze(1)], dim=1)  # [B, T+1]

            # EOS가 나온 샘플은 finished 표시
            finished = finished | (next_id == eos_id)

            # 모든 샘플이 EOS를 낸 경우 종료
            if torch.all(finished):
                break

        if outputs:
            return torch.stack(outputs, dim=1)  # [B, gen_len]
        return torch.empty((B, 0), dtype=torch.long, device=device)

    @torch.no_grad()
    def generate_with_beam_search(
        self,
        src: torch.Tensor,   # [B, S]
        max_len: int,
        bos_id: int,
        eos_id: int,
        src_pad_id: int,
        beam_size: int = 5,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """
        Beam Search decoding for better sequence generation.
        Developer log: Implements beam search with length penalty for improved accuracy.
        
        Args:
            src: [B, S] input sequence
            beam_size: number of beams to maintain
            length_penalty: penalty for longer sequences (higher = prefer longer)
        
        Returns:
            [B, T'] generated sequence (best beam)
        """
        device = src.device
        B, S = src.size()
        
        # --- Encoder (shared across all beams) ---
        src_emb = self.embed_in(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc_in(src_emb)
        src_key_padding_mask = (src == src_pad_id)
        memory = self.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, S, d_model]
        
        # For simplicity, process batch_size=1 at a time
        # (Full batched beam search is complex; this is a practical implementation)
        all_results = []
        
        for b in range(B):
            # Single sample beam search
            memory_b = memory[b:b+1]  # [1, S, d_model]
            src_mask_b = src_key_padding_mask[b:b+1]  # [1, S]
            
            # Initialize beams: [(sequence, score)]
            beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0)]
            
            for step in range(max_len):
                candidates = []
                
                for seq, score in beams:
                    # If sequence ended with EOS, keep it as is
                    if seq[0, -1].item() == eos_id:
                        candidates.append((seq, score))
                        continue
                    
                    # Decode one step
                    T = seq.size(1)
                    tgt_emb = self.embed_out(seq) * math.sqrt(self.d_model)  # [1, T, d_model]
                    tgt_emb = self.pos_enc_out(tgt_emb)
                    tgt_mask = _generate_square_subsequent_mask(T, device=device)
                    
                    dec_out = self.decoder(
                        tgt_emb,
                        memory_b,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=src_mask_b,
                    )  # [1, T, d_model]
                    
                    if self.use_digit_conv and self.digit_conv is not None:
                        conv_in = dec_out.transpose(1, 2)  # [1, C, T]
                        conv_out = self.digit_conv(conv_in)
                        conv_out = self.digit_conv_activation(conv_out)
                        dec_out = conv_out.transpose(1, 2)
                    
                    last_step = dec_out[:, -1, :]  # [1, d_model]
                    logits = self.out_proj(last_step)  # [1, out_vocab]
                    log_probs = torch.log_softmax(logits, dim=-1)  # [1, out_vocab]
                    
                    # Get top-k candidates
                    top_log_probs, top_indices = torch.topk(log_probs[0], beam_size)
                    
                    for log_prob, idx in zip(top_log_probs, top_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        # Score with length penalty
                        new_score = score + log_prob.item()
                        candidates.append((new_seq, new_score))
                
                # Select top beam_size candidates
                # Apply length penalty: score / (length ** length_penalty)
                scored_candidates = []
                for seq, score in candidates:
                    length = seq.size(1)
                    normalized_score = score / (length ** length_penalty)
                    scored_candidates.append((seq, score, normalized_score))
                
                scored_candidates.sort(key=lambda x: x[2], reverse=True)
                beams = [(seq, score) for seq, score, _ in scored_candidates[:beam_size]]
                
                # Early stopping if all beams end with EOS
                if all(seq[0, -1].item() == eos_id for seq, _ in beams):
                    break
            
            # Select best beam
            best_seq = beams[0][0][0, 1:]  # Remove BOS, [T']
            all_results.append(best_seq)
        
        # Pad results to same length
        max_result_len = max(seq.size(0) for seq in all_results)
        result_tensor = torch.full((B, max_result_len), eos_id, device=device, dtype=torch.long)
        for i, seq in enumerate(all_results):
            result_tensor[i, :seq.size(0)] = seq
        
        return result_tensor


# ========================
# InThon 규정용 Model
# ========================

class Model(BaseModel):
    """
    InThon Datathon 제출용 Model 클래스
    
    BaseModel을 상속받아 구현된 평가용 모델 래퍼입니다.
    - __init__에서 상대 경로로 best_model.pt 로드
    - predict(str) -> str 구현
    
    사용법:
        model = Model()  # 자동으로 best_model.pt 로드
        result = model.predict("12+34")  # "46" 반환
    """
    
    def __init__(self) -> None:
        """
        모델 초기화
        
        체크포인트를 로드하고 모델을 평가 모드로 설정합니다.
        모든 초기화는 이 메서드에서 완료되어야 합니다.
        """
        super().__init__()
        
        # 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 체크포인트 경로 (반드시 상대 경로 사용)
        CKPT_PATH = "best_model.pt"
        
        # 체크포인트 로드
        # 체크포인트는 dict 형식일 수 있으며, "model_state" 키가 있으면 사용,
        # 없으면 전체를 state_dict로 간주
        checkpoint = torch.load(CKPT_PATH, map_location=self.device)
        
        # 토크나이저 설정 로드 (체크포인트에 저장된 설정 필수)
        # tokenizer_config가 별도로 저장된 형식만 지원
        tokenizer_config_dict = checkpoint.get("tokenizer_config")
        if tokenizer_config_dict is None:
            raise ValueError(f"체크포인트에 'tokenizer_config'가 없습니다.")
        
        # 토크나이저 초기화 (체크포인트에서 로드한 설정 사용)
        input_chars = tokenizer_config_dict.get("input_chars", INPUT_CHARS)
        output_chars = tokenizer_config_dict.get("output_chars", OUTPUT_CHARS)
        add_special = tokenizer_config_dict.get("add_special", True)
        
        # 입력 토크나이저: 수식 문자 처리용
        self.input_tokenizer = CharTokenizer(
            input_chars if input_chars is not None else INPUT_CHARS,
            add_special=add_special,
        )
        # 출력 토크나이저: 숫자만 처리용
        self.output_tokenizer = CharTokenizer(
            output_chars if output_chars is not None else OUTPUT_CHARS,
            add_special=add_special,
        )
        
        # 모델 설정 로드 (체크포인트에 저장된 설정 필수)
        # model_config가 별도로 저장된 형식만 지원
        model_config_dict = checkpoint.get("model_config")
        if model_config_dict is None:
            raise ValueError(f"체크포인트에 'model_config'가 없습니다.")
        
        # Developer log: 제출용 Model 클래스에서는 RPN decoder를 사용하지 않음
        # model_config_dict에서 rpn_vocab을 제거하여 RPN decoder가 생성되지 않도록 함
        model_config_dict = model_config_dict.copy()  # 원본 수정 방지
        if "rpn_vocab" in model_config_dict:
            model_config_dict.pop("rpn_vocab")
        
        # TinySeq2Seq 모델 인스턴스 생성 (체크포인트에서 로드한 설정을 **kwargs로 전달)
        # self.model = TinySeq2Seq(
        #     in_vocab=self.input_tokenizer.vocab_size,  # 입력 vocab 크기
        #     out_vocab=self.output_tokenizer.vocab_size,  # 출력 vocab 크기
        #     **model_config_dict,  # 모델 설정을 **kwargs로 전달
        # ).to(self.device)  # 지정된 디바이스로 이동

        #Transformer 모델 인스턴스
        # Developer log: rpn_vocab=None으로 명시하여 RPN decoder 미생성 보장
        self.model = TransformerSeq2Seq(
        in_vocab=self.input_tokenizer.vocab_size,
        out_vocab=self.output_tokenizer.vocab_size,
        rpn_vocab=None,  # 제출용: RPN decoder 사용 안 함
        **model_config_dict,
        ).to(self.device)
        
        # 모델 가중치 로드
        # Developer log: checkpoint 구조 확인 및 model_state 추출
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        elif isinstance(checkpoint, dict):
            # checkpoint 자체가 state_dict인 경우
            model_state = checkpoint
        else:
            # checkpoint가 state_dict인 경우
            model_state = checkpoint
        
        # Developer log: RPN decoder는 학습 시에만 사용되며 inference에는 불필요
        # 제출용 Model 클래스에서는 RPN decoder가 없으므로 관련 키를 필터링
        if not isinstance(model_state, dict):
            raise ValueError(f"model_state must be a dict, got {type(model_state)}")
        
        # RPN 관련 키 제거 (rpn_decoder, rpn_embed, rpn_out, rpn_value_out)
        filtered_state = {
            k: v for k, v in model_state.items()
            if not k.startswith("rpn_")
        }
        
        # 필터링된 키 수 확인
        removed_keys = set(model_state.keys()) - set(filtered_state.keys())
        if removed_keys:
            print(f"⚠️ Removed {len(removed_keys)} RPN-related keys from checkpoint (inference에 불필요)")
            # 디버깅: 처음 5개 키만 출력
            sample_keys = list(removed_keys)[:5]
            print(f"   Sample removed keys: {sample_keys}")
        
        # strict=False로 설정하여 예상치 못한 키 무시 (방어적 처리)
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys in model (will use random init): {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    print(f"   Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    print(f"   Unexpected keys: {unexpected_keys}")
        except Exception as e:
            # 디버깅: 오류 발생 시 상세 정보 출력
            print(f"❌ Error loading state_dict: {e}")
            print(f"   Model state dict keys (first 10): {list(self.model.state_dict().keys())[:10]}")
            print(f"   Checkpoint keys (first 10): {list(filtered_state.keys())[:10]}")
            raise
        
        # 최대 생성 길이 설정
        self.max_len = 50
        
        # 평가 모드로 설정 (dropout, batch norm 등 비활성화)
        self.model.eval()
        
    def predict(self, input_text: str) -> str:
        """
        입력 수식을 받아 계산 결과를 반환
        
        Args:
            input_text: 계산할 수식 문자열 (예: "12+34", "5*6")
            
        Returns:
            계산 결과 문자열 (예: "46", "30")
            유효한 숫자가 아니면 빈 문자열 반환
        """
        # 입력 타입 검증 및 변환
        if not isinstance(input_text, str):
            input_text = str(input_text)
        
        # 배치 형식으로 변환 (target_text는 토크나이저 포맷을 맞추기 위한 dummy)
        batch = {"input_text": [input_text], "target_text": ["0"]}
        
        # 배치 토크나이징
        batch_tensors = tokenize_batch(batch, self.input_tokenizer, self.output_tokenizer)
        
        # 입력 텐서를 디바이스로 이동
        src = batch_tensors.src.to(self.device)
        
        # 추론 모드 (gradient 계산 비활성화)
        with torch.no_grad():
            # 시퀀스 생성
            gens = self.model.generate(
                src=src,  # 입력 시퀀스
                max_len=self.max_len,  # 최대 생성 길이
                bos_id=self.output_tokenizer.bos_id,  # 시작 토큰 ID
                eos_id=self.output_tokenizer.eos_id,  # 종료 토큰 ID
                src_pad_id=self.input_tokenizer.pad_id,  # 패딩 토큰 ID
            )
        
        # 생성된 토큰 ID를 문자열로 변환
        preds: List[str] = []
        for i in range(gens.size(0)):  # 배치 내 각 샘플에 대해
            seq_chars: List[str] = []
            
            for t in gens[i].tolist():  # 각 토큰 ID에 대해
                idx = int(t)
                
                # EOS 토큰을 만나면 즉시 중단
                if idx == self.output_tokenizer.eos_id:
                    break
                
                # 토큰 ID를 문자로 변환
                if idx in self.output_tokenizer.itos:
                    ch = self.output_tokenizer.itos[idx]
                    # 숫자만 추출 (특수 토큰 제외)
                    if ch.isdigit():
                        seq_chars.append(ch)
            
            preds.append("".join(seq_chars))
        
        # 첫 번째 예측 결과 반환 (배치 크기가 1이므로)
        pred = preds[0] if preds else ""
        
        # 유효한 숫자인지 검증 후 반환 (숫자가 아니면 빈 문자열)
        return pred if pred.isdigit() else ""
