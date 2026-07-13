# Security Policy

AxiomStack is a research and portfolio project, but it ships components that
warrant a clear security note.

## Reporting a vulnerability

Please open a private security advisory on GitHub, or email the maintainer listed
in the repository profile. Do not open a public issue for undisclosed
vulnerabilities. Expect an acknowledgement within a few days.

## Scope and non-goals

- **`python_exec` agent tool** runs code in a restricted namespace (no imports,
  no dunder access, whitelisted builtins, stdout captured). It is a *best-effort*
  guard for demos, **not** a security sandbox. Do not expose it to untrusted input
  in production; run untrusted code in a real sandbox (container/gVisor/WASM).
- **The FastAPI service** is intended for local use. Keep administrative
  ingestion, evaluation, and agent routes private; add authentication,
  authorization, and rate limiting before public deployment.
- **Model outputs** from the tiny GPT are not factual or safety-aligned at scale;
  the safety-refusal behavior is a benign, synthetic demonstration only.
- **Demo mode** uses a deterministic rule-based provider. The runtime and UI
  label it explicitly; it is not a DorkLLM capability claim.
- **Experiment tracking** is local by default. W&B mirroring is opt-in; do not
  place secrets, proprietary prompts, or private dataset contents in run configs
  or metric payloads.

## Data handling

The project uses only small public or synthetic datasets. No user data, secrets,
or credentials are collected or required. Do not commit real data or secrets;
`.env` files are git-ignored.
