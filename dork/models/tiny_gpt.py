"""``TinyGPT`` — a compact decoder-only transformer (GPT-2 architecture family).

The model is deliberately small (a few million parameters) and readable. It
demonstrates the full stack of LLM internals: token + positional embeddings,
stacked pre-norm transformer blocks with causal multi-head attention, weight
tying between the embedding and the output head, GPT-2-style initialization, and
autoregressive sampling. It is an *educational-scale* model — see
``docs/limitations.md`` for an honest discussion of scope.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from dork.models.layers import Block, KVCache, LayerNorm, RMSNorm, sinusoidal_position_embedding
from dork.utils.config import ModelConfig
from dork.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = get_logger(__name__)


class TinyGPT(nn.Module):
    """A small GPT-style language model.

    Args:
        config: A :class:`~dork.utils.config.ModelConfig`.
    """

    sinusoidal_pe: torch.Tensor  # registered buffer when pos_encoding == "sinusoidal"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.use_rope = config.pos_encoding == "rope"

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        drop_path_rates = torch.linspace(0.0, config.stochastic_depth, config.n_layer).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.n_embd,
                    config.n_head,
                    config.block_size,
                    config.dropout,
                    config.bias,
                    use_rope=self.use_rope,
                    norm_type=config.norm_type,
                    mlp_type=config.mlp_type,
                    stochastic_depth=float(drop_path_rates[i]),
                    n_kv_head=config.n_kv_head,
                    qk_norm=config.qk_norm,
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = (
            RMSNorm(config.n_embd)
            if config.norm_type == "rmsnorm"
            else LayerNorm(config.n_embd, bias=config.bias)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Positional information.
        if config.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_encoding == "sinusoidal":
            self.pos_emb = None
            self.register_buffer(
                "sinusoidal_pe",
                sinusoidal_position_embedding(config.block_size, config.n_embd),
                persistent=False,
            )
        else:  # rope — positions injected inside attention
            self.pos_emb = None

        # Weight tying: the input embedding and output projection share weights.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 §2.3).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        logger.info("Initialized TinyGPT with %.2fM parameters", self.num_params() / 1e6)

    # ── Initialization ───────────────────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Count parameters. With ``non_embedding=True``, exclude position embeddings."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    # ── Forward ──────────────────────────────────────────────────────
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and (optionally) the cross-entropy loss.

        Args:
            idx: Token ids of shape ``(B, T)`` with ``T <= block_size``.
            targets: Optional next-token targets of shape ``(B, T)``.

        Returns:
            ``(logits, loss)``. When ``targets`` is None, ``logits`` is computed
            only for the final position (an inference fast-path) and ``loss`` is None.
        """
        _, T = idx.shape
        if self.config.block_size < T:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}.")

        x = self.token_emb(idx)  # (B, T, n_embd)
        if self.pos_emb is not None:  # learned
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        elif hasattr(self, "sinusoidal_pe"):  # sinusoidal
            x = x + self.sinusoidal_pe[:T][None, :, :].to(x.dtype)
        x = self.drop(x)

        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        # Inference: only the last position is needed to predict the next token.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # ── Optimizer ────────────────────────────────────────────────────
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """Build AdamW with decay applied only to 2-D weight matrices.

        Biases, LayerNorm gains and embeddings are excluded from weight decay,
        following common practice for transformer training.
        """
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # Fused AdamW is a CUDA-only fast path.
        use_fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__doc__  # type: ignore[operator]
        extra = {"fused": True} if use_fused else {}
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        logger.info(
            "AdamW: %d decayed tensors, %d non-decayed tensors (fused=%s)",
            len(decay),
            len(no_decay),
            bool(extra),
        )
        return opt

    # ── Cached forward (incremental decoding) ─────────────────────────
    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Forward a (possibly single-token) chunk using a per-layer KV cache.

        Args:
            idx: Token ids of shape ``(B, T)``. During cached decoding ``T`` is
                typically 1 (the most recently sampled token).
            past: The previous step's list of per-layer ``(k, v)`` caches, or
                None on the first (prefill) call.

        Returns:
            ``(logits_last, present)`` where ``logits_last`` is ``(B, vocab)`` for
            the final position and ``present`` is the updated per-layer cache.
        """
        _, T = idx.shape
        past_len = past[0][0].shape[-2] if past is not None else 0
        total = past_len + T
        if total > self.config.block_size:
            raise ValueError(
                f"Cached sequence length {total} exceeds block_size {self.config.block_size}."
            )

        x = self.token_emb(idx)
        if self.pos_emb is not None:  # learned: offset positions by the cache length
            pos = torch.arange(past_len, past_len + T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        elif hasattr(self, "sinusoidal_pe"):
            x = x + self.sinusoidal_pe[past_len : past_len + T][None, :, :].to(x.dtype)
        x = self.drop(x)

        present: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            layer_past = past[i] if past is not None else None
            x, layer_present = block(x, layer_past=layer_past, use_cache=True)
            present.append(layer_present)  # type: ignore[arg-type]
        x = self.ln_f(x)
        logits = self.lm_head(x[:, -1, :])  # (B, vocab)
        return logits, present

    # ── Generation ───────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations.

        Two decode paths, numerically equivalent under greedy sampling:

        * ``use_cache=True`` (default): O(T) per step via a KV cache — prefill the
          prompt once, then feed only the newest token each step.
        * ``use_cache=False``: the simple O(T²) reference path that re-runs the
          full (cropped) context every step. Kept for clarity and as a test oracle.

        See :mod:`dork.generation.sampling` for the sampling primitives.
        """
        from dork.generation.sampling import sample_next_token

        self.eval()
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

        # Cached path: prefill on the (cropped) prompt, then step one token at a time.
        idx_cond = idx[:, -self.config.block_size :]
        logits, past = self.forward_with_cache(idx_cond)
        for _ in range(max_new_tokens):
            next_id = sample_next_token(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, next_id), dim=1)
            # If the cache would exceed the context window, drop it and re-prefill
            # on the cropped tail (rare for short generations; keeps correctness).
            if past[0][0].shape[-2] >= self.config.block_size:
                logits, past = self.forward_with_cache(idx[:, -self.config.block_size :])
            else:
                logits, past = self.forward_with_cache(next_id, past=past)
        return idx

    @classmethod
    def from_config(cls, config: ModelConfig) -> TinyGPT:
        """Convenience constructor."""
        return cls(config)
