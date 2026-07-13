"""Tests for the typed config layer."""

from __future__ import annotations

import pytest
from dork.utils.config import ModelConfig, TinyGPTConfig, load_tiny_gpt_config
from pydantic import ValidationError


def test_load_default_train_config():
    cfg = load_tiny_gpt_config("configs/train_tiny_gpt.yaml")
    assert cfg.model.n_embd % cfg.model.n_head == 0
    assert cfg.model.n_kv_head == cfg.model.n_head
    assert not cfg.model.qk_norm
    # vocab is synced from the tokenizer target.
    assert cfg.model.vocab_size == cfg.tokenizer.vocab_size


def test_load_frontier_train_config():
    cfg = load_tiny_gpt_config("configs/dorkllm_frontier.yaml")
    assert cfg.data.dataset == "tinystories"
    assert cfg.model.pos_encoding == "rope"
    assert cfg.model.norm_type == "rmsnorm"
    assert cfg.model.mlp_type == "swiglu"
    assert cfg.model.stochastic_depth > 0
    assert cfg.model.n_kv_head == 2
    assert cfg.model.n_head % cfg.model.n_kv_head == 0
    assert cfg.model.qk_norm
    assert cfg.training.gradient_accumulation_steps > 1


def test_model_config_rejects_indivisible_heads():
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=30, n_head=4)


@pytest.mark.parametrize("n_kv_head", [3, 16])
def test_model_config_rejects_invalid_kv_head_grouping(n_kv_head):
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=64, n_head=8, n_kv_head=n_kv_head)


def test_legacy_model_config_defaults_to_multi_head_attention():
    cfg = ModelConfig.model_validate(
        {"vocab_size": 64, "block_size": 32, "n_layer": 2, "n_head": 4, "n_embd": 64}
    )
    assert cfg.n_kv_head is None
    assert not cfg.qk_norm


def test_defaults_construct():
    cfg = TinyGPTConfig()
    assert cfg.seed == 1337
    assert cfg.training.max_steps > 0


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        ModelConfig(this_key_does_not_exist=1)
