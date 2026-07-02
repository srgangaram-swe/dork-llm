# Example Evaluation Report

This checked-in report mirrors the kind of artifact written to `reports/` by
`make eval`. It uses the deterministic mock provider so the project can produce
a full report offline in CI.

## Run Metadata

| Field | Value |
|---|---|
| Model | `mock-rulebased-v0` |
| Provider | `mock` |
| Seed | `1337` |
| Generated | `2026-06-28T20:25:12Z` |
| CI gate | PASS |
| Local tracking | `experiments/<run>/metadata.json`, `metrics.jsonl`, `summary.json` |

## Summary

| Suite | Category | N | Metric | Value |
|---|---|---:|---|---:|
| exact_match | reasoning | 10 | accuracy | 1.0000 |
| multiple_choice | reasoning | 10 | accuracy | 0.3000 |
| json_validity | structured_output | 8 | valid_rate | 1.0000 |
| instruction_following | instruction | 8 | constraint_pass_rate | 1.0000 |
| rag_faithfulness | retrieval | 8 | faithfulness | 0.7500 |
| tool_use | tool_use | 8 | tool_accuracy | 1.0000 |
| safety_refusal | safety | 10 | behavior_accuracy | 1.0000 |
| latency | performance | 1 | mean_ms | 0.0031 |

## Interpretation

The mock provider is strong on deterministic tasks it was built to handle:
arithmetic, JSON formatting, simple tool calls, and synthetic refusal behavior.
It is intentionally weak on multiple-choice because it defaults to `A`, which is
a useful reminder that a high score on one suite does not imply broad model
quality.

The RAG faithfulness suite passes most answerable cases but fails synthetic
unanswerable cases when the mock quotes context instead of refusing. This is an
honest failure mode and a good regression target for stronger grounded models.

## Representative Failures

### Multiple Choice

Prompt:

```text
In a transformer, what mechanism lets each token attend to others?
A) Convolution
B) Self-attention
C) Pooling
D) Dropout
Answer with the letter only.
```

Output: `A`

Expected: `B`

### RAG Faithfulness

Prompt category: unanswerable question with unrelated context.

Output:

```text
[1] The retriever returns the top-k most relevant chunks for a query. [1]
```

Expected behavior: refuse because the context did not contain the answer.

## Improvement Ideas

- Evaluate `local_gpt` after training instead of the mock provider.
- Add larger instruction and MCQ suites.
- Add entailment-based RAG faithfulness checks.
- Add citation precision/recall by claim.
- Track latency separately for generation, retrieval, reranking, and agent runs.
- Mirror selected metrics to W&B for longer-running comparison studies.
