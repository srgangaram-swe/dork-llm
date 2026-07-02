# Security Policy

Dork LLM is an educational/portfolio project, but it ships a few components that
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
- **The FastAPI service** ships without authentication and is intended for local
  use. Put it behind an auth proxy and rate limiting before any deployment.
- **Model outputs** from the tiny GPT are not factual or safety-aligned at scale;
  the safety-refusal behavior is a benign, synthetic demonstration only.
- **Experiment tracking** is local by default. W&B mirroring is opt-in; do not
  place secrets, proprietary prompts, or private dataset contents in run configs
  or metric payloads.

## Data handling

The project uses only small public or synthetic datasets. No user data, secrets,
or credentials are collected or required. Do not commit real data or secrets;
`.env` files are git-ignored.
