"""Tests for the tiny GPT model and sampling (require the [train] extra)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dork.generation.sampling import apply_top_k, apply_top_p, sample_next_token  # noqa: E402
from dork.models.layers import CausalSelfAttention, RMSNorm, SwiGLU  # noqa: E402
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
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
        pos_encoding="rope",
        norm_type="rmsnorm",
        mlp_type="swiglu",
        stochastic_depth=0.1,
        qk_norm=True,
    )
    model = TinyGPT(cfg)
    assert any(isinstance(m, RMSNorm) for m in model.modules())
    assert any(isinstance(m, SwiGLU) for m in model.modules())
    assert all(block.attn.n_kv_head == 2 for block in model.blocks)
    assert all(isinstance(block.attn.q_norm, RMSNorm) for block in model.blocks)

    x = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, loss = model(x, x)
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0

    model.eval()
    ref = model.generate(x[:1, :6], max_new_tokens=12, temperature=0.0, use_cache=False)
    cached = model.generate(x[:1, :6], max_new_tokens=12, temperature=0.0, use_cache=True)
    assert torch.equal(ref, cached)

    _, past = model.forward_with_cache(x[:1, :6])
    assert all(k.shape[1] == cfg.n_kv_head for k, _ in past)
    assert all(v.shape[1] == cfg.n_kv_head for _, v in past)


def test_grouped_query_attention_reduces_projection_and_cache_width():
    attn = CausalSelfAttention(n_embd=64, n_head=8, block_size=32, n_kv_head=2)
    assert attn.c_attn.out_features == 64 + 2 * 2 * 8

    x = torch.randn(2, 7, 64)
    out, present = attn(x, use_cache=True)
    assert out.shape == x.shape
    assert present is not None
    key, value = present
    assert key.shape == value.shape == (2, 2, 7, 8)


def test_legacy_multi_head_state_dict_shape_remains_compatible():
    from dork.utils.config import ModelConfig

    legacy_payload = {
        "vocab_size": 64,
        "block_size": 16,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "dropout": 0.0,
    }
    original = TinyGPT(ModelConfig.model_validate(legacy_payload))
    restored = TinyGPT(ModelConfig.model_validate(legacy_payload))
    restored.load_state_dict(original.state_dict(), strict=True)

    assert restored.blocks[0].attn.n_kv_head == 4
    assert restored.blocks[0].attn.c_attn.out_features == 3 * 64


@pytest.mark.parametrize("n_kv_head,qk_norm", [(4, False), (2, False), (2, True)])
def test_cached_multi_token_chunk_matches_full_prefill(n_kv_head, qk_norm):
    from dork.utils.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=96,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=n_kv_head,
        n_embd=64,
        dropout=0.0,
        pos_encoding="rope",
        qk_norm=qk_norm,
    )
    model = TinyGPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 11))

    full_logits, _ = model.forward_with_cache(idx)
    _, prefix_cache = model.forward_with_cache(idx[:, :4])
    chunk_logits, chunk_cache = model.forward_with_cache(idx[:, 4:], past=prefix_cache)

    assert torch.allclose(chunk_logits, full_logits, atol=1e-6, rtol=1e-5)
    assert all(key.shape == (1, n_kv_head, 11, 16) for key, _ in chunk_cache)


def test_cached_multi_token_chunk_matches_full_prefill_without_flash():
    from dork.utils.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        block_size=24,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
        pos_encoding="rope",
        qk_norm=True,
    )
    model = TinyGPT(cfg).eval()
    for block in model.blocks:
        block.attn._flash = False
    idx = torch.randint(0, cfg.vocab_size, (1, 10))

    full_logits, _ = model.forward_with_cache(idx)
    _, prefix_cache = model.forward_with_cache(idx[:, :3])
    chunk_logits, _ = model.forward_with_cache(idx[:, 3:], past=prefix_cache)

    assert torch.allclose(chunk_logits, full_logits, atol=1e-6, rtol=1e-5)


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
