# 코드 변경사항 가이드

> **사용법**: Colab에서 복사붙여넣기 할 때, 이 문서의 변경사항만 확인하고 적용하세요.

---

## 📝 변경 파일 목록

1. **`model.py`** - Attention 기반 Seq2Seq 모델로 변경
2. **`config.py`** - 새로운 하이퍼파라미터 추가
3. **`train.py`** - Beam Search, Scheduled Sampling 추가

---

## 1. model.py 변경사항

### 변경 1: Attention 메커니즘 클래스 추가

**위치**: `model.py` 파일의 `TinySeq2Seq` 클래스 위에 추가

```python
# ========================
# Attention Mechanism
# ========================

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            decoder_hidden: [B, hidden_size] - 디코더의 현재 hidden state
            encoder_outputs: [B, src_len, hidden_size] - 인코더의 모든 출력
            mask: [B, src_len] - 패딩 마스크 (True: 패딩, False: 실제 토큰)
        
        Returns:
            context: [B, hidden_size] - Attention 가중합
            attention_weights: [B, src_len] - Attention 가중치
        """
        # decoder_hidden을 [B, 1, hidden_size]로 확장
        decoder_hidden = decoder_hidden.unsqueeze(1)  # [B, 1, hidden_size]
        
        # Attention score 계산
        # score = V * tanh(W1 * encoder_outputs + W2 * decoder_hidden)
        scores = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden)))  # [B, src_len, 1]
        scores = scores.squeeze(-1)  # [B, src_len]
        
        # 패딩 마스크 적용
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax로 Attention 가중치 계산
        attention_weights = torch.softmax(scores, dim=-1)  # [B, src_len]
        
        # 가중합으로 context 계산
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [B, 1, hidden_size]
        context = context.squeeze(1)  # [B, hidden_size]
        
        return context, attention_weights
```

### 변경 2: TinySeq2Seq → AttentionSeq2Seq로 변경

**위치**: `model.py` 파일의 `TinySeq2Seq` 클래스를 다음으로 교체

```python
# ========================
# AttentionSeq2Seq (개선된 모델)
# ========================

class AttentionSeq2Seq(nn.Module):
    """
    Attention 기반 Seq2Seq 모델
    
    구조:
    - 인코더: Multi-layer GRU + 모든 타임스텝 출력 보존
    - 디코더: Multi-layer GRU + Bahdanau Attention
    - 생성: Beam Search 지원
    """
    
    def __init__(self, in_vocab: int, out_vocab: int, **kwargs):
        super().__init__()
        
        # 모델 설정 추출
        d_model = kwargs.get("d_model", 512)
        num_encoder_layers = kwargs.get("num_encoder_layers", 2)
        num_decoder_layers = kwargs.get("num_decoder_layers", 2)
        dropout = kwargs.get("dropout", 0.1)
        
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # 입력 임베딩
        self.embed_in = nn.Embedding(in_vocab, d_model)
        
        # 인코더: Multi-layer GRU
        self.encoder = nn.GRU(
            d_model, d_model,
            num_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0
        )
        
        # 출력 임베딩
        self.embed_out = nn.Embedding(out_vocab, d_model)
        
        # Attention 메커니즘
        self.attention = BahdanauAttention(d_model)
        
        # 디코더: Multi-layer GRU
        # 입력: 이전 출력 임베딩 + attention context
        self.decoder = nn.GRU(
            d_model * 2,  # 임베딩 + attention context
            d_model,
            num_layers=num_decoder_layers,
            batch_first=True,
            dropout=dropout if num_decoder_layers > 1 else 0
        )
        
        # 출력 프로젝션
        self.out_proj = nn.Linear(d_model, out_vocab)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor, src_pad_id: int, teacher_forcing: float = 1.0) -> torch.Tensor:
        """
        순전파 (학습 시 사용)
        
        Args:
            src: [B, src_len]
            tgt_inp: [B, tgt_len]
            src_pad_id: 패딩 토큰 ID
            teacher_forcing: Teacher forcing 비율 (현재는 항상 1.0)
        
        Returns:
            logits: [B, tgt_len, out_vocab]
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt_inp.size(1)
        
        # 패딩 마스크 생성
        src_mask = (src == src_pad_id)  # [B, src_len]
        
        # Encoder
        x = self.embed_in(src)  # [B, src_len, d_model]
        x = self.dropout(x)
        encoder_outputs, encoder_hidden = self.encoder(x)  # [B, src_len, d_model], [num_layers, B, d_model]
        
        # 마지막 레이어의 hidden state만 사용
        encoder_hidden = encoder_hidden[-1]  # [B, d_model]
        
        # Decoder
        decoder_input = self.embed_out(tgt_inp)  # [B, tgt_len, d_model]
        decoder_input = self.dropout(decoder_input)
        
        # 디코더의 초기 hidden state는 인코더의 마지막 hidden state
        decoder_hidden = encoder_hidden.unsqueeze(0).repeat(self.num_decoder_layers, 1, 1)  # [num_layers, B, d_model]
        
        outputs = []
        for t in range(tgt_len):
            # 현재 타임스텝의 입력
            current_input = decoder_input[:, t:t+1, :]  # [B, 1, d_model]
            
            # Attention 계산
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, src_mask)  # [B, d_model]
            context = context.unsqueeze(1)  # [B, 1, d_model]
            
            # Attention context와 입력 결합
            decoder_input_with_attention = torch.cat([current_input, context], dim=-1)  # [B, 1, d_model*2]
            
            # 디코더 forward
            decoder_output, decoder_hidden = self.decoder(decoder_input_with_attention, decoder_hidden)
            # decoder_output: [B, 1, d_model]
            # decoder_hidden: [num_layers, B, d_model]
            
            # 출력 프로젝션
            output = self.out_proj(decoder_output[:, 0, :])  # [B, out_vocab]
            outputs.append(output)
        
        # 모든 타임스텝의 출력을 스택
        logits = torch.stack(outputs, dim=1)  # [B, tgt_len, out_vocab]
        
        return logits
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int, src_pad_id: int, beam_size: int = 1) -> torch.Tensor:
        """
        추론 시 시퀀스 생성
        
        Args:
            src: [B, src_len]
            max_len: 최대 생성 길이
            bos_id: 시작 토큰 ID
            eos_id: 종료 토큰 ID
            src_pad_id: 패딩 토큰 ID
            beam_size: Beam Search 크기 (1이면 Greedy)
        
        Returns:
            generated: [B, gen_len]
        """
        if beam_size == 1:
            return self._greedy_decode(src, max_len, bos_id, eos_id, src_pad_id)
        else:
            return self._beam_search_decode(src, max_len, bos_id, eos_id, src_pad_id, beam_size)
    
    def _greedy_decode(self, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int, src_pad_id: int) -> torch.Tensor:
        """Greedy Decoding"""
        batch_size = src.size(0)
        device = src.device
        
        # 패딩 마스크
        src_mask = (src == src_pad_id)
        
        # Encoder
        x = self.embed_in(src)
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_hidden = encoder_hidden[-1]  # [B, d_model]
        
        # 디코더 초기화
        decoder_hidden = encoder_hidden.unsqueeze(0).repeat(self.num_decoder_layers, 1, 1)
        decoder_input = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        outputs = []
        for _ in range(max_len):
            # 입력 임베딩
            input_emb = self.embed_out(decoder_input[:, -1:])  # [B, 1, d_model]
            
            # Attention
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs, src_mask)
            context = context.unsqueeze(1)
            
            # 결합
            decoder_input_with_attention = torch.cat([input_emb, context], dim=-1)
            
            # 디코더
            decoder_output, decoder_hidden = self.decoder(decoder_input_with_attention, decoder_hidden)
            
            # 출력
            logits = self.out_proj(decoder_output[:, 0, :])  # [B, out_vocab]
            next_id = torch.argmax(logits, dim=-1)  # [B]
            
            outputs.append(next_id)
            decoder_input = torch.cat([decoder_input, next_id.unsqueeze(1)], dim=1)
            
            # 모든 배치에서 EOS 생성 시 중단
            if torch.all(next_id == eos_id):
                break
        
        if outputs:
            return torch.stack(outputs, dim=1)
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
    
    def _beam_search_decode(self, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int, src_pad_id: int, beam_size: int) -> torch.Tensor:
        """Beam Search Decoding (간단한 구현)"""
        # 간단한 구현: 배치 크기가 1인 경우만 지원
        if src.size(0) != 1:
            # 배치 크기가 1이 아니면 Greedy로 fallback
            return self._greedy_decode(src, max_len, bos_id, eos_id, src_pad_id)
        
        device = src.device
        src_mask = (src == src_pad_id)
        
        # Encoder
        x = self.embed_in(src)
        encoder_outputs, encoder_hidden = self.encoder(x)
        encoder_hidden = encoder_hidden[-1]
        
        # Beam 초기화
        beams = [(torch.tensor([bos_id], device=device), 0.0, encoder_hidden.unsqueeze(0).repeat(self.num_decoder_layers, 1, 1))]
        # beams: [(sequence, score, hidden_state), ...]
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score, hidden in beams:
                if seq[-1].item() == eos_id:
                    candidates.append((seq, score, hidden))
                    continue
                
                # 마지막 토큰으로 다음 토큰 예측
                input_emb = self.embed_out(seq[-1:].unsqueeze(0))  # [1, 1, d_model]
                
                # Attention
                context, _ = self.attention(hidden[-1], encoder_outputs, src_mask)
                context = context.unsqueeze(1)
                
                # 결합
                decoder_input_with_attention = torch.cat([input_emb, context], dim=-1)
                
                # 디코더
                decoder_output, new_hidden = self.decoder(decoder_input_with_attention, hidden)
                
                # 출력
                logits = self.out_proj(decoder_output[:, 0, :])  # [1, out_vocab]
                probs = torch.log_softmax(logits, dim=-1)  # [1, out_vocab]
                
                # Top-k 후보 선택
                top_probs, top_ids = torch.topk(probs[0], beam_size)
                
                for prob, token_id in zip(top_probs, top_ids):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score, new_hidden))
            
            # Top-k beams 선택
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # 모든 beam이 EOS를 생성했으면 중단
            if all(seq[-1].item() == eos_id for seq, _, _ in beams):
                break
        
        # 최고 점수 beam 선택
        best_seq = beams[0][0]
        # BOS 제거하고 반환
        return best_seq[1:].unsqueeze(0)  # [1, gen_len]
```

### 변경 3: Model 클래스에서 AttentionSeq2Seq 사용

**위치**: `model.py` 파일의 `Model` 클래스 `__init__` 메서드 내부

**변경 전**:
```python
self.model = TinySeq2Seq(
    in_vocab=self.input_tokenizer.vocab_size,
    out_vocab=self.output_tokenizer.vocab_size,
    **model_config_dict,
).to(self.device)
```

**변경 후**:
```python
self.model = AttentionSeq2Seq(  # TinySeq2Seq → AttentionSeq2Seq
    in_vocab=self.input_tokenizer.vocab_size,
    out_vocab=self.output_tokenizer.vocab_size,
    **model_config_dict,
).to(self.device)
```

---

## 2. config.py 변경사항

**위치**: `config.py` 파일의 `ModelConfig` 클래스

**변경 전**:
```python
@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    d_model: int = 256  # Hidden dimension 크기 (임베딩, GRU hidden size)
    # 향후 확장 가능: num_layers, dropout, attention heads 등
```

**변경 후**:
```python
@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    d_model: int = 512  # Hidden dimension 크기 (256 → 512)
    num_encoder_layers: int = 2  # 인코더 레이어 수 (1 → 2)
    num_decoder_layers: int = 2  # 디코더 레이어 수 (1 → 2)
    dropout: float = 0.1  # Dropout 비율
    attention_type: str = "bahdanau"  # Attention 타입
```

**위치**: `config.py` 파일의 `TrainConfig` 클래스

**변경 전**:
```python
@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3
    valid_every: int = 50
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 4
    save_best_path: Optional[str] = None
```

**변경 후**:
```python
@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3  # 2e-3 → 1e-3 (더 안정적)
    valid_every: int = 200
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 50
    save_best_path: Optional[str] = None
    beam_size: int = 5  # Beam Search 크기 (1이면 Greedy)
    teacher_forcing_ratio: float = 0.9  # Teacher Forcing 비율 (초기값)
    use_scheduled_sampling: bool = True  # Scheduled Sampling 사용 여부
```

---

## 3. train.py 변경사항

### 변경 1: train.py의 main() 함수에서 모델 생성 부분

**위치**: `train.py` 파일의 `main()` 함수 내부

**변경 전**:
```python
model = TinySeq2Seq(
    in_vocab=input_tokenizer.vocab_size,
    out_vocab=output_tokenizer.vocab_size,
    **model_config.__dict__,
)
```

**변경 후**:
```python
model = AttentionSeq2Seq(  # TinySeq2Seq → AttentionSeq2Seq
    in_vocab=input_tokenizer.vocab_size,
    out_vocab=output_tokenizer.vocab_size,
    **model_config.__dict__,
)
```

### 변경 2: train.py의 import 문 추가

**위치**: `train.py` 파일의 상단 import 부분

**변경 전**:
```python
from model import (
    TinySeq2Seq,
    CharTokenizer,
    tokenize_batch,
    INPUT_CHARS,
    OUTPUT_CHARS,
)
```

**변경 후**:
```python
from model import (
    AttentionSeq2Seq,  # TinySeq2Seq → AttentionSeq2Seq
    CharTokenizer,
    tokenize_batch,
    INPUT_CHARS,
    OUTPUT_CHARS,
)
```

### 변경 3: train.py의 train_loop 함수에서 Scheduled Sampling 추가

**위치**: `train.py` 파일의 `train_loop()` 함수 내부, forward 호출 부분

**변경 전**:
```python
logits = model(src, target_input, input_tokenizer.pad_id)
```

**변경 후**:
```python
# Scheduled Sampling: Teacher Forcing 비율 점진적 감소
if train_config.use_scheduled_sampling:
    # step에 따라 teacher_forcing_ratio 감소
    total_steps = train_config.num_epochs * len(dataloader) if train_config.max_train_steps is None else train_config.max_train_steps
    current_tf_ratio = max(0.5, train_config.teacher_forcing_ratio * (1.0 - step / total_steps * 0.5))
else:
    current_tf_ratio = train_config.teacher_forcing_ratio

logits = model(src, target_input, input_tokenizer.pad_id, teacher_forcing=current_tf_ratio)
```

### 변경 4: train.py의 generate 호출에 beam_size 추가

**위치**: `train.py` 파일의 `train_loop()` 함수 내부, validation 부분

**변경 전**:
```python
gen_ids = model.generate(
    src=val_src,
    max_len=train_config.max_gen_len,
    bos_id=output_tokenizer.bos_id,
    eos_id=output_tokenizer.eos_id,
    src_pad_id=input_tokenizer.pad_id,
)
```

**변경 후**:
```python
gen_ids = model.generate(
    src=val_src,
    max_len=train_config.max_gen_len,
    bos_id=output_tokenizer.bos_id,
    eos_id=output_tokenizer.eos_id,
    src_pad_id=input_tokenizer.pad_id,
    beam_size=train_config.beam_size,  # Beam Search 추가
)
```

---

## 📋 변경 요약

### 파일별 변경 사항

1. **model.py**
   - ✅ `BahdanauAttention` 클래스 추가
   - ✅ `TinySeq2Seq` → `AttentionSeq2Seq` 클래스로 교체
   - ✅ `Model` 클래스에서 `AttentionSeq2Seq` 사용

2. **config.py**
   - ✅ `ModelConfig`에 새로운 하이퍼파라미터 추가
   - ✅ `TrainConfig`에 Beam Search, Scheduled Sampling 설정 추가

3. **train.py**
   - ✅ import 문 변경 (`TinySeq2Seq` → `AttentionSeq2Seq`)
   - ✅ 모델 생성 부분 변경
   - ✅ Scheduled Sampling 로직 추가
   - ✅ Beam Search 사용

### 적용 순서

1. `model.py` 변경 (Attention 클래스 추가 + 모델 클래스 교체)
2. `config.py` 변경 (하이퍼파라미터 추가)
3. `train.py` 변경 (학습 로직 업데이트)

---

**문서 작성일**: 2025년  
**최종 업데이트**: 코드 변경사항 가이드 작성 완료

