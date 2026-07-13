# Limitations

AxiomStack is a compact language-model systems project. Its value is the
inspectable connection among model math, statistical evidence, and full-stack
delivery—not a claim that DorkLLM is a frontier model or that DorkChat is ready
for unsupervised public use.

## Model scale and behavior

DorkLLM has millions, not billions, of parameters and trains on small public
corpora. Expected weaknesses include:

- repetition and malformed text;
- weak long-range coherence;
- unreliable factual knowledge and instruction following;
- sensitivity to seed, data, and optimization choices;
- no broad safety alignment.

Sampling temperature and retrieval scores are not calibrated confidence
probabilities. The UI must not present them that way.

## Artifacts and chat readiness

Checkpoints and tokenizers are generated locally and ignored by git. A fresh
clone therefore has no trained DorkLLM until the documented training commands
run or a compatible artifact is mounted. Strict service mode reports not-ready
instead of silently replacing the model. Explicit demo mode uses a deterministic
rule-based provider and labels it as a demo.

The browser-to-model path demonstrates integration; it does not turn the small
from-scratch model into a general assistant. RAG can improve grounding only when
retrieval finds relevant evidence and the configured generator follows the
grounded prompt.

## Data

The default corpus is Tiny Shakespeare or a bundled public-domain fallback. The
modern-small path caps locally downloaded TinyStories input by default. These
data do not cover modern factual knowledge, robust instructions, coding,
multilingual behavior, or representative production traffic.

No LANL, employer, classified, proprietary, private, or sensitive data is used
or required.

## Statistical evidence

Current evaluation datasets are small and largely synthetic. They are useful
for deterministic regression testing, not deployment certification.

The committed scaling study changes multiple architectural factors across a
small number of configurations and uses a single seed. Its fitted trend is
descriptive, not a validated scaling law. The v0.3 milestone explicitly tracks
compute-matched designs, repeated seeds, paired uncertainty, calibration,
risk-coverage analysis, and retrieval confidence intervals.

The pre-v0.2 SFT comparison is not valid because the original label construction
used same-position rather than next-token targets. v0.2 fixes the objective and
tests; corrected quality numbers require a new artifact and fixed evaluation
protocol.

## Retrieval and agents

The default hash embedder and memory vector store are deterministic test
backends with limited semantic quality. Citation extraction verifies marker-to-
chunk mapping, not that every generated claim is entailed by its citation.

The research agent uses bounded deterministic routing. Its local Python tool is
not a hardened security boundary and must not be exposed to untrusted public
users.

## Serving and deployment

The service is a local research runtime. It has in-process model state and
metrics and no durable conversation store. A public deployment still needs the
authentication, authorization, rate limiting, persistent telemetry, supply-
chain scanning, load validation, and rollback work tracked in the roadmap.

Administrative ingestion, evaluation, and agent routes should remain local or
protected until those controls exist.

## What the repository does demonstrate

- explicit transformer, attention, normalization, and causal-objective code;
- tested training, post-training, checkpoint, and inference paths;
- an extensible evaluation and experiment framework;
- source-provenance retrieval and bounded tools;
- typed API/service boundaries and an accessible browser client;
- Python and browser tests, CI, a non-root container, and documented delivery
  governance.

It does not yet demonstrate frontier pretraining, broad post-training,
production-scale distributed training, calibrated deployment readiness, or a
security-reviewed public service.
