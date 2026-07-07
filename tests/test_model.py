"""Tests for the tiny GPT model and sampling (require the [train] extra)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dork.generation.sampling import apply_top_k, apply_top_p, sample_next_token  # noqa: E402
from dork.models.layers import RMSNorm, SwiGLU  # noqa: E402
from dork.models.tiny_gpt import TinyGPT  # noqa: E402

pytestmark = pytest.mark.torch


def test_forward_shapes(tiny_model_config):
    model = TinyGPT(tiny_model_config)
    x = torch.randint(0, tiny_model_config.vocab_size, (2, tiny_model_config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, tiny_model_config.block_size, tiny_model_config.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_inference_fastpath_returns_last_position(tiny_model_config):
    model = TinyGPT(tiny_model_config)
    x = torch.randint(0, tiny_model_config.vocab_size, (1, 5))
    logits, loss = model(x)
    assert logits.shape == (1, 1, tiny_model_config.vocab_size)
    assert loss is None


def test_weight_tying(tiny_model_config):
    model = TinyGPT(tiny_model_config)
    assert model.lm_head.weight is model.token_emb.weight


def test_generate_extends_sequence(tiny_model_config):
    model = TinyGPT(tiny_model_config)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=7, temperature=1.0, top_k=5)
    assert out.shape == (1, 10)


def test_block_size_guard(tiny_model_config):
    model = TinyGPT(tiny_model_config)
    too_long = torch.zeros((1, tiny_model_config.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model(too_long)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "rope"])
def test_positional_variants_forward(pos):
    from dork.utils.config import ModelConfig

    cfg = ModelConfig(
        vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, pos_encoding=pos
    )
    model = TinyGPT(cfg)
    x = torch.randint(0, 64, (1, 16))
    logits, _ = model(x)
    assert logits.shape[-1] == 64


def test_frontier_architecture_variants_forward_and_cache():
    from dork.utils.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=128,
        block_size=48,
        n_layer=3,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        pos_encoding="rope",
        norm_type="rmsnorm",
        mlp_type="swiglu",
        stochastic_depth=0.1,
    )
    model = TinyGPT(cfg)
    assert any(isinstance(m, RMSNorm) for m in model.modules())
    assert any(isinstance(m, SwiGLU) for m in model.modules())

    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, loss = model(x, x)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0

    model.eval()
    ref = model.generate(x[:1, :6], max_new_tokens=12, temperature=0.0, use_cache=False)
    cached = model.generate(x[:1, :6], max_new_tokens=12, temperature=0.0, use_cache=True)
    assert torch.equal(ref, cached)


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "rope"])
def test_kv_cache_matches_reference(pos):
    """Greedy decoding must be numerically identical with and without the KV cache."""
    from dork.utils.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=96, block_size=64, n_layer=3, n_head=4, n_embd=64, dropout=0.0, pos_encoding=pos
    )
    model = TinyGPT(cfg)
    model.eval()
    idx = torch.randint(0, 96, (1, 8))
    ref = model.generate(idx, max_new_tokens=24, temperature=0.0, use_cache=False)
    cached = model.generate(idx, max_new_tokens=24, temperature=0.0, use_cache=True)
    assert torch.equal(ref, cached)


def test_kv_cache_crosses_context_window(tiny_model_config):
    """Generating beyond block_size must not crash (cache is re-prefilled)."""
    model = TinyGPT(tiny_model_config)
    model.eval()
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=tiny_model_config.block_size * 2, temperature=0.0)
    assert out.shape[1] == 4 + tiny_model_config.block_size * 2


def test_greedy_sampling_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert sample_next_token(logits, temperature=0.0).item() == 1


def test_top_k_masks_low_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    masked = apply_top_k(logits.clone(), top_k=2)
    assert torch.isinf(masked[0, 0]) and torch.isinf(masked[0, 1])
    assert not torch.isinf(masked[0, 3])


def test_top_p_keeps_top_token():
    logits = torch.tensor([[10.0, 1.0, 0.5]])
    masked = apply_top_p(logits.clone(), top_p=0.5)
    assert not torch.isinf(masked[0, 0])
