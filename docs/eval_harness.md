# Evaluation Harness

## Purpose

The evaluation harness models the kind of pre-deployment measurement layer used
around real LLM systems. It can evaluate a local Tiny GPT checkpoint, a Hugging
Face model, or the deterministic mock provider used for offline CI.

The goal is not to declare a model "good" with one score. The goal is to expose
tradeoffs across language modeling, task correctness, structured output,
retrieval grounding, tool use, safety behavior, and latency.

## Entry Points

```bash
make eval
dork eval --config configs/eval_default.yaml
python scripts/evaluate_model.py --config configs/eval_default.yaml
```

Outputs are written to `reports/` by default:

- `eval_report.json`: machine-readable full report.
- `eval_summary.csv`: flat summary table.
- `eval_report.md`: human-facing report.
- `eval_metrics.png`: optional plot when matplotlib is installed.

`reports/` is ignored by git. A representative checked-in example lives at
`docs/example_eval_report.md`.

## Providers

Configured under `model` in `configs/eval_default.yaml`:

- `mock`: deterministic rule-based provider for CI and demos.
- `local_gpt`: loads `artifacts/tiny_gpt/ckpt.pt`.
- `hf`: loads an open Hugging Face causal LM when `transformers` is installed.

The provider contract is `LanguageModel.complete(...)` plus optional
`perplexity(...)`.

## Suites

| Suite | Category | What It Measures |
|---|---|---|
| `perplexity` | language_modeling | Token-level language modeling loss on a held-out corpus. |
| `exact_match` | reasoning | Short-answer arithmetic and normalized exact/contains match. |
| `multiple_choice` | reasoning | Letter extraction and MCQ accuracy. |
| `json_validity` | structured_output | JSON parse rate, required-key coverage, schema pass rate. |
| `instruction_following` | instruction | Verifiable constraints such as length, prefix, include/exclude text. |
| `rag_faithfulness` | retrieval | Citation coverage, grounding overlap, refusal on unanswerable prompts. |
| `tool_use` | tool_use | Tool selection and JSON call reliability. |
| `safety_refusal` | safety | Benign synthetic refusal/compliance behavior. |
| `latency` | performance | Mean, p50, p95 latency and request throughput. |

Datasets are small JSONL fixtures under `dork/evaluation/datasets/`. They are
synthetic and public-safe.

## CI Gating

`report.thresholds` maps `suite.metric` to a minimum acceptable value. The
harness records each check and a top-level `gate.passed` boolean. CI can fail on
the gate or simply publish it as a status artifact depending on how strict you
want the workflow to be.

Example:

```yaml
report:
  thresholds:
    exact_match.accuracy: 0.8
    safety_refusal.behavior_accuracy: 0.8
```

## Design Notes

- Evaluators are registered by name, so new suites can be added without changing
  the orchestration loop.
- The mock provider deliberately has uneven strengths: strong arithmetic and
  JSON, weak multiple-choice. This makes sample reports honest rather than
  artificially perfect.
- RAG faithfulness is heuristic and lightweight. It is useful for regression
  tests, not a replacement for human review or stronger entailment models.
- Latency numbers for the mock provider are not representative of real model
  serving. Use `local_gpt` or `hf` for meaningful performance work.
