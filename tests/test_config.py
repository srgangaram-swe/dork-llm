"""Tests for the typed config layer."""

from __future__ import annotations

import pytest
from dork.utils.config import ModelConfig, TinyGPTConfig, load_tiny_gpt_config
from pydantic import ValidationError


def test_load_default_train_config():
    cfg = load_tiny_gpt_config("configs/train_tiny_gpt.yaml")
    assert cfg.model.n_embd % cfg.model.n_head == 0
    # vocab is synced from the tokenizer target.
    assert cfg.model.vocab_size == cfg.tokenizer.vocab_size


def test_model_config_rejects_indivisible_heads():
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=30, n_head=4)


def test_defaults_construct():
    cfg = TinyGPTConfig()
    assert cfg.seed == 1337
    assert cfg.training.max_steps > 0


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        ModelConfig(this_key_does_not_exist=1)
