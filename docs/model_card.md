# Model Card: Dork Tiny GPT

## Summary

Dork Tiny GPT is a compact decoder-only transformer implemented from scratch in
PyTorch. It is intentionally educational-scale: it demonstrates the core
engineering patterns behind GPT-style language models without claiming frontier
capability.

## Architecture

- Decoder-only transformer in the GPT-2 family.
- Token embeddings plus configurable positional encoding: learned, sinusoidal,
  or RoPE.
- Pre-norm transformer blocks.
- Multi-head causal self-attention with PyTorch scaled dot-product attention
  when available.
- Incremental KV-cache decoding for fast inference, with a reference decode path
  used for numerical parity tests and speed benchmarks.
- GELU MLP with 4x hidden expansion.
- Residual connections, LayerNorm, dropout, AdamW, gradient clipping, cosine LR
  schedule, checkpointing, and weight tying.
- Sampling supports temperature, greedy decoding, top-k, and top-p.

Default config: `configs/train_tiny_gpt.yaml`.

## Intended Use

This model is intended to show that the author can:

- implement a transformer from first principles;
- train and checkpoint a small causal language model;
- post-train it with supervised instruction tuning (SFT);
- evaluate perplexity and generation behavior;
- benchmark KV-cache generation speedups;
- expose the model through a reusable generation provider;
- serve it through CLI, scripts, FastAPI, and Streamlit.

It is useful for portfolio demonstration, tests, local experiments, and
educational analysis. It is not intended for factual QA, production generation,
policy decisions, or user-facing autonomous behavior.

## Training Data

The default pipeline uses public text only:

- Tiny Shakespeare when network access is available.
- A bundled public-domain Shakespeare excerpt as an offline fallback.
- Custom public or synthetic text can be configured through
  `data.dataset: custom` and `data.raw_path`.

No employer data, sensitive documents, private APIs, or proprietary corpora are
required or included.

## Local Baseline

One local run in this workspace used a smaller training profile and produced:

| Field | Value |
|---|---:|
| Parameters | 3,705,088 |
| Vocabulary | 2,048 |
| Training tokens | 388,613 |
| Training time | 1.49 minutes |
| Final train loss | 4.5095 |
| Final validation loss | 4.6829 |
| Train perplexity on 4k chars | 99.69 |

The local checkpoint and tokenizer are ignored by git. Regenerate them with:

```bash
make train-tokenizer
make train-small-gpt
make sft
make benchmark
make generate
```

## Known Behavior

The model learns local Shakespeare-like syntax and short stylistic continuations.
With limited data and training budget, it also produces malformed words,
repeated phrases, and hallucinated names. This is expected and documented rather
than hidden: the project is about the end-to-end LLM systems stack, not a high
quality standalone model.

## Evaluation

Use:

```bash
make eval
make benchmark_inference
make scaling-study
make experiments
```

The evaluation harness can target the local checkpoint, a Hugging Face model, or
the deterministic mock provider used in CI. See `docs/eval_harness.md`.

The scaling study in `scripts/scaling_study.py` is a small ablation over model
width/depth that writes `reports/scaling_study.json` and the committed plot at
`docs/assets/scaling_study.png`. It is useful for demonstrating methodology, not
for making frontier-scale claims.

## Risks and Mitigations

- Factuality risk: the model is not trained for factual QA. Use RAG with
  citations for grounded answers.
- Safety risk: the model is not alignment-trained. The eval harness includes
  benign synthetic refusal tests, but this is not a safety certification.
- Overclaiming risk: all docs state that the model is compact and
  educational-scale.
- Reproducibility risk: configs, seeds, checkpoint metadata, local experiment
  tracking, and ignored artifact rules keep runs repeatable without committing
  large binaries.
