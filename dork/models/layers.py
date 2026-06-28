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

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, n_head, T, head_dim)
        seq_len = q.shape[-2]
        cos, sin = self._cos_sin(seq_len, q.device, q.dtype)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with a single fused QKV projection."""

    causal_mask: torch.Tensor  # registered buffer on the non-flash fallback path

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # Fused projection for q, k, v then an output projection.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None
        self._flash = hasattr(F, "scaled_dot_product_attention")
        if not self._flash:
            # Lower-triangular causal mask for the fallback path.
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self._flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # pragma: no cover - exercised only on old torch
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


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
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias, use_rope)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
