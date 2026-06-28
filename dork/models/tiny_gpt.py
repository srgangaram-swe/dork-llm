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

from dork.models.layers import Block, LayerNorm, sinusoidal_position_embedding
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
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.n_embd,
                    config.n_head,
                    config.block_size,
                    config.dropout,
                    config.bias,
                    use_rope=self.use_rope,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
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
            x = block(x)
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

    # ── Generation ───────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations.

        See :mod:`dork.generation.sampling` for the sampling primitives; this
        method wires them into the decode loop with context-window cropping.
        """
        from dork.generation.sampling import sample_next_token

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            next_id = sample_next_token(logits, temperature, top_k, top_p)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    @classmethod
    def from_config(cls, config: ModelConfig) -> TinyGPT:
        """Convenience constructor."""
        return cls(config)
