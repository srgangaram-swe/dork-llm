"""Typed configuration models (pydantic v2) loaded from YAML.

The tiny-GPT pipeline is strongly typed end-to-end so a malformed config fails
loudly at load time rather than deep inside a training loop. Eval and RAG configs
are validated but permissive (``extra="allow"``) to keep them easy to extend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dork.utils.io import load_yaml


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    dataset: Literal["tiny_shakespeare", "tinystories", "wikitext2", "custom"] = "tiny_shakespeare"
    data_dir: str = "data"
    raw_path: str = "data/raw/tiny_shakespeare.txt"
    val_fraction: float = Field(0.1, gt=0.0, lt=0.9)


class TokenizerConfig(_Base):
    type: Literal["bpe", "char"] = "bpe"
    vocab_size: int = Field(4096, ge=16)
    path: str = "tokenizers/tiny_gpt_bpe.json"
    special_tokens: list[str] = Field(
        default_factory=lambda: ["<|endoftext|>", "<|pad|>", "<|unk|>"]
    )


class ModelConfig(_Base):
    vocab_size: int = Field(4096, ge=16)
    block_size: int = Field(256, ge=8)
    n_layer: int = Field(6, ge=1)
    n_head: int = Field(6, ge=1)
    n_embd: int = Field(384, ge=8)
    dropout: float = Field(0.1, ge=0.0, le=0.9)
    bias: bool = False
    pos_encoding: Literal["learned", "sinusoidal", "rope"] = "learned"

    @model_validator(mode="after")
    def _check_divisible(self) -> ModelConfig:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        return self


class TrainingConfig(_Base):
    batch_size: int = Field(32, ge=1)
    max_steps: int = Field(2000, ge=1)
    gradient_accumulation_steps: int = Field(1, ge=1)
    eval_interval: int = Field(250, ge=1)
    eval_iters: int = Field(100, ge=1)
    log_interval: int = Field(50, ge=1)
    learning_rate: float = Field(3.0e-4, gt=0.0)
    weight_decay: float = Field(0.1, ge=0.0)
    beta1: float = Field(0.9, gt=0.0, lt=1.0)
    beta2: float = Field(0.95, gt=0.0, lt=1.0)
    grad_clip: float = Field(1.0, ge=0.0)
    decay_lr: bool = True
    warmup_steps: int = Field(100, ge=0)
    min_lr: float = Field(3.0e-5, ge=0.0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    compile: bool = False
    out_dir: str = "artifacts/tiny_gpt"
    always_save_checkpoint: bool = False


class GenerationConfig(_Base):
    max_new_tokens: int = Field(200, ge=1)
    temperature: float = Field(0.8, ge=0.0)
    top_k: int | None = Field(50, ge=0)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)


class SFTConfig(_Base):
    """Supervised fine-tuning (post-training) settings."""

    base_out_dir: str = "artifacts/tiny_gpt"  # checkpoint to fine-tune from
    out_dir: str = "artifacts/tiny_gpt_sft"
    n_arith: int = Field(48, ge=0)
    val_fraction: float = Field(0.2, gt=0.0, lt=0.9)
    batch_size: int = Field(16, ge=1)
    max_steps: int = Field(300, ge=1)
    eval_interval: int = Field(50, ge=1)
    learning_rate: float = Field(1.0e-4, gt=0.0)
    warmup_steps: int = Field(20, ge=0)
    min_lr: float = Field(1.0e-5, ge=0.0)


class TinyGPTConfig(_Base):
    """Aggregate config for the full tiny-GPT pipeline."""

    seed: int = 1337
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)

    @model_validator(mode="after")
    def _sync_vocab(self) -> TinyGPTConfig:
        # Keep the model vocab in sync with the tokenizer target.
        self.model.vocab_size = self.tokenizer.vocab_size
        return self


class PermissiveConfig(BaseModel):
    """Base for eval/rag configs: validated but open to extension."""

    model_config = ConfigDict(extra="allow")


class EvalConfig(PermissiveConfig):
    seed: int = 1337
    model: dict[str, Any] = Field(default_factory=dict)
    suites: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    datasets_dir: str = "dork/evaluation/datasets"


class RagConfig(PermissiveConfig):
    seed: int = 1337
    ingest: dict[str, Any] = Field(default_factory=dict)
    embeddings: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, Any] = Field(default_factory=dict)


def load_tiny_gpt_config(path: str | Path) -> TinyGPTConfig:
    """Load and validate a tiny-GPT training config from YAML."""
    return TinyGPTConfig.model_validate(load_yaml(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load and validate an evaluation config from YAML."""
    return EvalConfig.model_validate(load_yaml(path))


def load_rag_config(path: str | Path) -> RagConfig:
    """Load and validate a RAG config from YAML."""
    return RagConfig.model_validate(load_yaml(path))
