"""Transformer building blocks for the tiny GPT.

Each block is implemented from first principles (no ``nn.TransformerEncoderLayer``)
so the internals — causal masking, multi-head attention, residual stream and
pre-layernorm — are explicit and inspectable. Attention uses PyTorch's fused
``scaled_dot_product_attention`` (FlashAttention kernels) when available and falls
back to an explicit, masked-softmax implementation otherwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (GPT-2 style: ``bias=False``)."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by many modern decoder LLMs."""

    def __init__(self, ndim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate the variance in fp32 for numerical stability, then restore
        # the activation dtype so Q/K normalization remains compatible with V
        # under bf16/fp16 autocast.
        input_dtype = x.dtype
        compute_dtype = (
            torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
        )
        x_compute = x.to(compute_dtype)
        rms = torch.rsqrt(x_compute.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.to(compute_dtype) * x_compute * rms).to(input_dtype)


def sinusoidal_position_embedding(block_size: int, n_embd: int) -> torch.Tensor:
    """Return the classic (Vaswani et al.) fixed sinusoidal position table."""
    position = torch.arange(block_size).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
    pe = torch.zeros(block_size, n_embd)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) applied to queries and keys."""

    inv_freq: torch.Tensor  # registered buffer (annotated for the type-checker)

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype):
        # ``offset`` shifts positions so incremental (KV-cached) decoding rotates
        # each new token by its *absolute* position, not its position in the step.
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, n_head, T, head_dim)
        seq_len = q.shape[-2]
        cos, sin = self._cos_sin(seq_len, offset, q.device, q.dtype)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


#: A per-layer cache entry. Keys and values are compactly stored as
#: ``(B, n_kv_head, T_kv, head_dim)`` and expanded only for attention compute.
KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional grouped-query attention.

    Supports incremental decoding via an optional key/value cache: pass the
    previous step's ``layer_past`` and set ``use_cache=True`` to reuse already-
    computed keys/values instead of recomputing attention over the whole prefix.
    With ``n_kv_head < n_head``, several query heads share each key/value head and
    the cache remains compact. Optional per-head QK RMS normalization stabilizes
    attention logits without changing the residual-stream normalization.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = False,
        n_kv_head: int | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        n_kv_head = n_head if n_kv_head is None else n_kv_head
        if n_kv_head < 1:
            raise ValueError("n_kv_head must be positive")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.dropout = dropout

        # A single projection remains checkpoint-compatible with standard MHA:
        # when n_kv_head == n_head, its shape is the original 3 * n_embd.
        self.c_attn = nn.Linear(n_embd, n_embd + 2 * self.kv_dim, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.q_norm: nn.Module = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None
        self._flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        layer_past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split((self.n_embd, self.kv_dim, self.kv_dim), dim=2)
        # Queries use n_head; keys/values retain only n_kv_head for GQA.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Number of already-cached positions (0 during prefill / training).
        offset = layer_past[0].shape[-2] if layer_past is not None else 0
        if self.rope is not None:
            q, k = self.rope(q, k, offset=offset)

        # Prepend the cached keys/values so this step attends to the full prefix.
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present: KVCache | None = (k, v) if use_cache else None

        # Expand compact KV heads only for the attention kernel, never in cache.
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            attn_k = k.repeat_interleave(repeats, dim=1)
            attn_v = v.repeat_interleave(repeats, dim=1)
        else:
            attn_k, attn_v = k, v

        # Fused SDPA's built-in non-square causal mask is upper-left aligned, so
        # it is incorrect for a multi-token chunk appended to a cached prefix.
        # Use an absolute-position mask for that case; preserve the fast causal
        # kernel for ordinary prefill and no mask for a one-token decode step.
        attn_mask: torch.Tensor | None = None
        is_causal = T > 1 and offset == 0
        if T > 1 and offset > 0:
            q_pos = torch.arange(offset, offset + T, device=x.device).view(T, 1)
            k_pos = torch.arange(attn_k.shape[-2], device=x.device).view(1, -1)
            attn_mask = k_pos <= q_pos

        if self._flash:
            y = F.scaled_dot_product_attention(
                q,
                attn_k,
                attn_v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:  # pragma: no cover - exercised only on old torch
            att = (q @ attn_k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask, float("-inf"))
            elif is_causal:
                causal = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
                att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ attn_v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    """Position-wise feedforward network with a 4x hidden expansion and GELU."""

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLU(nn.Module):
    """Gated feedforward network with SwiGLU activation.

    The hidden width follows the common 8/3 expansion used to keep parameter
    count near a 4x GELU MLP while adding a learned gate.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        hidden_dim = max(8, int(8 * n_embd / 3))
        self.c_fc = nn.Linear(n_embd, 2 * hidden_dim, bias=bias)
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        return self.dropout(self.c_proj(value * F.silu(gate)))


class StochasticDepth(nn.Module):
    """Drop whole residual branches during training, scaled to preserve expectation."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("stochastic depth probability must be in [0, 1).")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class Block(nn.Module):
    """A pre-norm transformer block: ``x + attn(ln1(x))`` then ``x + mlp(ln2(x))``."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = False,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        stochastic_depth: float = 0.0,
        n_kv_head: int | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.ln_1: nn.Module
        self.ln_2: nn.Module
        self.mlp: nn.Module

        if norm_type == "layernorm":
            self.ln_1 = LayerNorm(n_embd, bias=bias)
            self.ln_2 = LayerNorm(n_embd, bias=bias)
        elif norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(n_embd)
            self.ln_2 = RMSNorm(n_embd)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size,
            dropout,
            bias,
            use_rope,
            n_kv_head=n_kv_head,
            qk_norm=qk_norm,
        )
        if mlp_type == "gelu":
            self.mlp = MLP(n_embd, dropout, bias)
        elif mlp_type == "swiglu":
            self.mlp = SwiGLU(n_embd, dropout, bias)
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")
        self.drop_path = StochasticDepth(stochastic_depth)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x, present
