# Model card: DorkLLM

## Summary

DorkLLM is AxiomStack's compact decoder-only transformer family, implemented
from explicit PyTorch components. It has a laptop-friendly reference profile and
a modern-small research profile. Both are educational-scale models intended for
architecture, training, inference, and evaluation experiments—not frontier or
general-assistant claims.

## Architecture

- Decoder-only, pre-normalized transformer.
- Learned, sinusoidal, or rotary positional information.
- LayerNorm or RMSNorm residual-stream normalization.
- GELU MLP or SwiGLU gated feed-forward blocks.
- Multi-head attention or grouped-query attention with a compact KV cache.
- Optional per-head QK RMS normalization.
- Fused PyTorch scaled-dot-product attention when available and an explicit
  masked-softmax reference path.
- Offset-aware causal masks for cached multi-token chunks.
- Weight-tied token embedding/output projection.
- Optional per-layer stochastic depth.
- AdamW, gradient clipping, cosine decay, warmup, gradient accumulation,
  autocast, checkpointing, and optional compilation.

Default profile: `configs/train_tiny_gpt.yaml`.

Modern-small profile: `configs/dorkllm_frontier.yaml`. The filename is retained
for compatibility; the documentation deliberately avoids calling this a
frontier model.

## Objectives

Pretraining uses causal next-token cross-entropy. Supervised fine-tuning uses
the same one-token shift while masking prompt and padding targets, so only the
response and end-of-text targets contribute to loss. Long examples preserve
response supervision by trimming prompt context before response targets.

The v0.2 audit found that the earlier SFT dataset aligned each target with its
same-position input. That made the earlier post-training comparison invalid.
The implementation and regression tests now enforce true next-token alignment;
old SFT checkpoints must be retrained before they are used as evidence.

## Profiles

| Setting | Baseline | Modern-small |
|---|---:|---:|
| Context | 128 | 512 |
| Layers | 4 | 8 |
| Width | 256 | 512 |
| Query heads | 4 | 8 |
| KV heads | 4 | 2 |
| Position | learned | RoPE |
| Normalization | LayerNorm | RMSNorm + QK RMSNorm |
| Feed-forward | GELU | SwiGLU |
| Stochastic depth | 0 | 0.05 |

Grouped-query attention keeps eight query heads while storing only two KV heads
in the modern-small cache. Unit tests assert cache shape, projection size,
legacy multi-head checkpoint compatibility, and numerical parity with full
causal prefill.

## Intended use

DorkLLM is intended to demonstrate and test:

- transformer architecture and causal-objective implementation;
- local pretraining and post-training mechanics;
- checkpoint and tokenizer compatibility;
- sampling and incremental inference;
- architecture ablations and performance measurement;
- typed model integration through CLI, API, RAG, and DorkChat.

It is not intended for factual QA, high-stakes decisions, autonomous action,
safety-critical use, or unsupervised public deployment. DorkChat exposes the
active provider and degradation state so a deterministic demo provider cannot
be mistaken for a trained DorkLLM.

## Training data

The default path uses public text only:

- Tiny Shakespeare when download is available;
- a bundled public-domain excerpt as the offline fallback;
- configurable public or synthetic custom text.

The modern-small configuration can use a capped TinyStories sample for local
research. It does not train on the full corpus by default and should not be
described as a TinyStories-scale result. No employer, classified, proprietary,
or sensitive data is required or included.

## Historical local baseline

A pre-v0.2 baseline run recorded the following pretraining measurements:

| Field | Value |
|---|---:|
| Parameters | 3,705,088 |
| Vocabulary | 2,048 |
| Training tokens | 388,613 |
| Training time | 1.49 minutes |
| Final train loss | 4.5095 |
| Final validation loss | 4.6829 |
| Train perplexity on 4k characters | 99.69 |

These values describe one local run, not a population estimate. The checkpoint
is ignored by git. No corrected-SFT quality number is claimed until a new fixed
split, seed, artifact hash, and paired report are published.

## Runtime resolution

The service validates checkpoint/tokenizer vocabulary compatibility. It reports
the requested provider, active provider, artifact, device, and any degradation
reason. Strict mode fails readiness when no compatible model exists. A mock is
available only through explicit demo mode and is labeled as such.

Candidate local artifacts are considered from the explicitly requested path,
then modern-small SFT, modern-small base, baseline SFT, and baseline base paths.
Selection order is not a quality claim; experiment metadata must justify a
candidate before release promotion.

## Evaluation

```bash
make eval
make benchmark
make scaling-study
make experiments
```

The suite covers perplexity where supported, task behavior, structured output,
retrieval faithfulness, citation behavior, safety fixtures, tool selection, and
latency. The current datasets are deliberately small. The v0.3 roadmap adds
paired inference, repeated seeds, calibration, selective prediction, and
controlled architecture/scaling experiments.

## Known behavior and risks

- Small-corpus generations are repetitive, locally coherent at best, and often
  malformed.
- The model does not contain reliable factual knowledge or robust instruction
  following.
- No broad safety alignment has been performed.
- Sampling controls change variability but do not create calibrated
  confidence.
- Hash-based retrieval and small evaluation fixtures are test backends, not
  production validation.
- Model and tokenizer artifacts are local and must be reconstructed or mounted.

See [`limitations.md`](limitations.md) for system-level limitations and
[`github_issues_plan.md`](github_issues_plan.md) for the measured improvement
roadmap.
