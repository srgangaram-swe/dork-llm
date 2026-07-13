# AxiomStack delivery roadmap

This document mirrors the live GitHub roadmap for
[`srgangaram-swe/dork-llm`](https://github.com/srgangaram-swe/dork-llm). The
GitHub milestones and issues are authoritative; this file explains the intent
and dependency order without embedding a script that can create duplicates.

## Product names

- **AxiomStack** is the platform and portfolio project.
- **DorkLLM** is the from-scratch decoder model family.
- **DorkChat** is the browser research cockpit.
- **Proof. Probability. Production.** is the project thesis: mathematical
  correctness, statistically defensible evidence, and production-quality
  delivery belong in the same system.

## Branch and release flow

```text
short-lived branch -> dev -> main -> prod
```

- `dev` is the integration branch and the target for feature pull requests.
- `main` is the stable public portfolio branch.
- `prod` is the deployed release pointer.
- Promotions use pull requests in that order. They are never cherry-picked.
- A short-lived branch is deleted only after its tip is verified as an ancestor
  of `dev`.

## v0.2 — DorkChat vertical slice

Due 2026-07-24. This is the current sprint. Its definition of done is a
truthful, tested browser-to-model path with the correctness defects found in the
initial audit repaired.

- [#2 — Correct causal SFT, bounded token F1, and cached-attention invariants](https://github.com/srgangaram-swe/dork-llm/issues/2)
- [#3 — Modernize DorkLLM with grouped-query attention and QK normalization](https://github.com/srgangaram-swe/dork-llm/issues/3)
- [#4 — Add explicit model runtime resolution and truthful readiness metadata](https://github.com/srgangaram-swe/dork-llm/issues/4)
- [#5 — Ship a bounded streaming DorkChat API contract](https://github.com/srgangaram-swe/dork-llm/issues/5)
- [#6 — Build the accessible AxiomStack DorkChat research cockpit](https://github.com/srgangaram-swe/dork-llm/issues/6)
- [#7 — Close full-stack CI, container, documentation, and branch-governance gaps](https://github.com/srgangaram-swe/dork-llm/issues/7)

## v0.3 — Statistical rigor and uncertainty

Due 2026-08-21. This phase adds reusable paired inference, controlled
multi-seed studies, calibration and selective prediction, uncertainty-aware IR
evaluation, and an accessible experiment comparison surface. See
[issues #8–#12](https://github.com/srgangaram-swe/dork-llm/issues?q=is%3Aissue%20milestone%3A%22AxiomStack%20v0.3%3A%20Statistical%20Rigor%20%26%20Uncertainty%22).

## v0.4 — Deep learning systems

Due 2026-09-25. This phase covers compute-matched architecture ablations, exact
resume and multi-device training, LoRA, preference optimization, and measured
quantization/distillation tradeoffs. See
[issues #13–#17](https://github.com/srgangaram-swe/dork-llm/issues?q=is%3Aissue%20milestone%3A%22AxiomStack%20v0.4%3A%20Deep%20Learning%20Systems%22).

## v0.5 — Grounded production platform

Due 2026-10-30. This phase evolves retrieval, agents, persistence,
observability, security, supply-chain controls, and deployment. See
[issues #18–#23](https://github.com/srgangaram-swe/dork-llm/issues?q=is%3Aissue%20milestone%3A%22AxiomStack%20v0.5%3A%20Grounded%20Production%20Platform%22).

## v1.0 — Reproducible public release

Due 2026-12-04. The release phase publishes reconstruction manifests, model and
data cards, a research-style benchmark report, the portfolio case study, and a
provenance-backed release. See
[issues #24–#27](https://github.com/srgangaram-swe/dork-llm/issues?q=is%3Aissue%20milestone%3A%22AxiomStack%20v1.0%3A%20Reproducible%20Public%20Release%22).

Every issue is assigned to `srgangaram-swe` and carries type, area, priority,
and effort labels. Acceptance criteria live on the issue so implementation and
review use the same definition of done.
