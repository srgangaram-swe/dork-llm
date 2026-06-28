# GitHub Issues Plan

Status checked on 2026-06-28:

- Remote: `https://github.com/srgangaram-swe/dork-llm`
- Default branch: `main`
- Repository visibility: public
- Remote commits: none visible before the first local publish
- Issues: none
- Milestones: none
- Labels: only GitHub defaults
- Local `gh` status: installed but not authenticated

Because the local GitHub CLI session was not authenticated, this file provides a
ready-to-run bootstrap script. Run it after:

```bash
gh auth login
gh auth status
```

## Bootstrap Commands

Copy this script into your shell from the repository root. It is idempotent for
labels and creates the planned milestones/issues.

```bash
set -euo pipefail

REPO="srgangaram-swe/dork-llm"

gh label create architecture -R "$REPO" --color "5319e7" --description "Architecture and system design" --force
gh label create data -R "$REPO" --color "0e8a16" --description "Datasets, ingestion, and preprocessing" --force
gh label create tokenizer -R "$REPO" --color "1d76db" --description "Tokenizer training and loading" --force
gh label create model -R "$REPO" --color "0052cc" --description "Model architecture and inference" --force
gh label create training -R "$REPO" --color "006b75" --description "Training loop, optimization, and checkpoints" --force
gh label create evaluation -R "$REPO" --color "fbca04" --description "Evaluation suites, metrics, and reports" --force
gh label create rag -R "$REPO" --color "0e8a16" --description "Retrieval augmented generation" --force
gh label create agents -R "$REPO" --color "bfdadc" --description "Agent tools and orchestration" --force
gh label create api -R "$REPO" --color "1d76db" --description "FastAPI service" --force
gh label create dashboard -R "$REPO" --color "c2e0c6" --description "Streamlit dashboard" --force
gh label create testing -R "$REPO" --color "d4c5f9" --description "Tests and smoke coverage" --force
gh label create docs -R "$REPO" --color "0075ca" --description "Documentation" --force
gh label create devops -R "$REPO" --color "f9d0c4" --description "CI, Docker, tooling, and release workflows" --force
gh label create portfolio -R "$REPO" --color "b60205" --description "Portfolio, resume, and launch polish" --force
gh label create enhancement -R "$REPO" --color "a2eeef" --description "New feature or improvement" --force
gh label create bug -R "$REPO" --color "d73a4a" --description "Bug or regression" --force

gh api -X POST "repos/$REPO/milestones" -f title="Milestone 1: Project Architecture and Tooling" -f description="Repository structure, tooling, configs, Docker, and CI." || true
gh api -X POST "repos/$REPO/milestones" -f title="Milestone 2: Tiny GPT Training Pipeline" -f description="Tokenizer, dataset preparation, transformer model, training, and generation." || true
gh api -X POST "repos/$REPO/milestones" -f title="Milestone 3: Evaluation Harness" -f description="Reusable evaluation suites, reports, gates, and comparisons." || true
gh api -X POST "repos/$REPO/milestones" -f title="Milestone 4: RAG and Agentic Research Assistant" -f description="Document ingestion, retrieval, citations, and tools." || true
gh api -X POST "repos/$REPO/milestones" -f title="Milestone 5: API, Dashboard, and Demo" -f description="FastAPI, Streamlit, smoke demos, and benchmarks." || true
gh api -X POST "repos/$REPO/milestones" -f title="Milestone 6: Documentation and Portfolio Polish" -f description="README, design docs, model card, limitations, resume, and LinkedIn materials." || true

issue () {
  local title="$1"
  local milestone="$2"
  local labels="$3"
  local goal="$4"
  local files="$5"
  local acceptance="$6"

  gh issue create -R "$REPO" \
    --title "$title" \
    --milestone "$milestone" \
    --label "$labels" \
    --body "$(cat <<BODY
## Goal
$goal

## Implementation Notes
Keep changes modular, typed, tested, and local-first. Avoid large generated artifacts and sensitive data.

## Relevant Files
$files

## Acceptance Criteria
$acceptance
BODY
)"
}

issue "Scaffold Dork LLM repository structure" \
  "Milestone 1: Project Architecture and Tooling" \
  "architecture,portfolio,enhancement" \
  "Create a coherent package, apps, scripts, configs, tests, docs, and sample data layout." \
  "dork/, apps/, scripts/, configs/, tests/, docs/, data/sample_docs/" \
  "- Package imports cleanly\n- CLI/scripts share orchestration\n- README explains the system layout"

issue "Add development tooling, Makefile, Docker, and CI" \
  "Milestone 1: Project Architecture and Tooling" \
  "devops,testing,enhancement" \
  "Provide reproducible local and CI workflows for install, lint, typecheck, tests, smoke runs, and Docker." \
  "pyproject.toml, Makefile, Dockerfile, .pre-commit-config.yaml, .github/workflows/ci.yml" \
  "- make help lists expected commands\n- CI runs lint, mypy, tests, and smoke train\n- Docker image runs the API"

issue "Implement dataset preparation pipeline" \
  "Milestone 2: Tiny GPT Training Pipeline" \
  "data,training,enhancement" \
  "Prepare public/local text corpora with an offline fallback and no sensitive data." \
  "dork/data/datasets.py, scripts/prepare_dataset.py, data/README.md" \
  "- Tiny Shakespeare downloads when available\n- fallback corpus works offline\n- generated raw data stays ignored"

issue "Implement tokenizer training and loading utilities" \
  "Milestone 2: Tiny GPT Training Pipeline" \
  "tokenizer,training,enhancement" \
  "Support byte-level BPE and character tokenizers behind one interface." \
  "dork/tokenizer/, scripts/train_tokenizer.py" \
  "- BPE trains with tokenizers extra\n- char tokenizer works without heavy deps\n- save/load round trips pass tests"

issue "Implement decoder-only transformer architecture in PyTorch" \
  "Milestone 2: Tiny GPT Training Pipeline" \
  "model,training,enhancement" \
  "Build a compact GPT-style model from explicit PyTorch components." \
  "dork/models/layers.py, dork/models/tiny_gpt.py" \
  "- forward pass returns logits/loss\n- causal masking prevents future attention\n- positional variants are tested"

issue "Implement training loop with checkpointing and validation" \
  "Milestone 2: Tiny GPT Training Pipeline" \
  "training,model,enhancement" \
  "Train Tiny GPT with validation, LR scheduling, gradient clipping, and checkpoints." \
  "dork/training/, scripts/train_tiny_gpt.py" \
  "- smoke training writes a reloadable checkpoint\n- training history records train/val loss\n- config controls runtime"

issue "Implement text generation with top-k, top-p, and temperature sampling" \
  "Milestone 2: Tiny GPT Training Pipeline" \
  "model,training,enhancement" \
  "Generate continuations with greedy, temperature, top-k, and top-p controls." \
  "dork/generation/, scripts/generate_text.py, dork/serving/service.py" \
  "- sampler functions are unit-tested\n- CLI/API pass sampling controls through\n- trained model generates text locally"

issue "Add perplexity and generation-quality evaluation" \
  "Milestone 3: Evaluation Harness" \
  "evaluation,model,enhancement" \
  "Measure language-model quality with perplexity and qualitative sample outputs." \
  "dork/evaluation/evaluators.py, dork/generation/generator.py, docs/example_eval_report.md" \
  "- perplexity suite runs against compatible providers\n- report captures failures\n- sample generations are documented honestly"

issue "Build reusable LLM evaluation harness" \
  "Milestone 3: Evaluation Harness" \
  "evaluation,testing,enhancement" \
  "Create a registry-based harness that runs suites, aggregates metrics, and writes reports." \
  "dork/evaluation/harness.py, dork/evaluation/base.py, dork/evaluation/report.py" \
  "- JSON/CSV/Markdown reports are written\n- thresholds produce gate checks\n- tests cover harness output"

issue "Add structured-output and JSON-validity evaluations" \
  "Milestone 3: Evaluation Harness" \
  "evaluation,enhancement" \
  "Evaluate parseable JSON, required-key coverage, and schema pass rate." \
  "dork/evaluation/evaluators.py, dork/evaluation/datasets/json_tasks.jsonl" \
  "- JSON validity and key coverage metrics are reported\n- invalid outputs become failed cases\n- dataset is synthetic/public-safe"

issue "Add RAG retrieval faithfulness and citation checks" \
  "Milestone 3: Evaluation Harness" \
  "evaluation,rag,enhancement" \
  "Measure citation coverage, grounding overlap, and refusal on unanswerable questions." \
  "dork/evaluation/evaluators.py, dork/evaluation/datasets/rag_faithfulness.jsonl" \
  "- answerable cases require valid citations\n- unanswerable cases require refusal\n- report includes failure examples"

issue "Implement document ingestion and chunking" \
  "Milestone 4: RAG and Agentic Research Assistant" \
  "rag,data,enhancement" \
  "Load Markdown/text/PDF documents and chunk them with source offsets." \
  "dork/rag/loaders.py, dork/rag/chunking.py, data/sample_docs/" \
  "- chunks retain source and character offsets\n- multiple chunking strategies are tested\n- sample docs ingest offline"

issue "Implement embedding and local vector database indexing" \
  "Milestone 4: RAG and Agentic Research Assistant" \
  "rag,data,enhancement" \
  "Index chunks with deterministic hash embeddings and optional heavier backends." \
  "dork/rag/embeddings.py, dork/rag/vectorstore.py" \
  "- hash embeddings are deterministic\n- memory store returns sorted hits\n- generated vector stores are ignored"

issue "Implement source-grounded RAG query pipeline" \
  "Milestone 4: RAG and Agentic Research Assistant" \
  "rag,evaluation,enhancement" \
  "Retrieve, rerank, answer with citations, and refuse without evidence." \
  "dork/rag/pipeline.py, scripts/query_rag.py" \
  "- answers map citation markers to source chunks\n- min-score threshold can force refusal\n- tests cover cited answer and refusal"

issue "Implement agentic research assistant tools" \
  "Milestone 4: RAG and Agentic Research Assistant" \
  "agents,rag,enhancement" \
  "Add bounded agent workflows for search, summarize, compare, extract claims, plan experiments, and calculate." \
  "dork/agents/research_agent.py, dork/agents/tools.py, scripts/run_agent.py" \
  "- intent routing is deterministic and tested\n- tool steps are recorded\n- structured outputs and citations are returned"

issue "Add FastAPI service endpoints" \
  "Milestone 5: API, Dashboard, and Demo" \
  "api,enhancement" \
  "Serve generation, evaluation, RAG, agent, health, and metrics endpoints." \
  "apps/api.py, dork/serving/" \
  "- endpoints return pydantic responses\n- service falls back to mock model\n- API smoke tests pass"

issue "Add Streamlit dashboard" \
  "Milestone 5: API, Dashboard, and Demo" \
  "dashboard,api,enhancement" \
  "Provide a usable UI for generation, evaluation, RAG, agent, and metrics workflows." \
  "apps/dashboard.py" \
  "- dashboard has tabs for core workflows\n- RAG citations are visible\n- metrics render without a separate API server"

issue "Add benchmark scripts for latency and throughput" \
  "Milestone 5: API, Dashboard, and Demo" \
  "evaluation,devops,enhancement" \
  "Measure local generation latency and throughput for the trained Tiny GPT." \
  "scripts/benchmark_inference.py, dork/pipelines.py, Makefile" \
  "- make benchmark and make benchmark_inference work\n- output includes mean, p50, p95, and tokens/sec\n- command is documented"

issue "Add unit tests and integration tests" \
  "Milestone 5: API, Dashboard, and Demo" \
  "testing,enhancement" \
  "Cover model, tokenizer, data, eval, RAG, agent, API, and smoke workflows." \
  "tests/, scripts/smoke_test.py" \
  "- pytest passes locally\n- slow training test is marked\n- CI runs a smoke training path"

issue "Write architecture, model card, and design docs" \
  "Milestone 6: Documentation and Portfolio Polish" \
  "docs,architecture,portfolio" \
  "Document system architecture, model card, evaluation design, RAG design, agent design, and limitations." \
  "docs/" \
  "- README links resolve\n- limitations are explicit\n- docs are suitable for public review"

issue "Write README with quickstart and examples" \
  "Milestone 6: Documentation and Portfolio Polish" \
  "docs,portfolio" \
  "Create a comprehensive README with architecture diagram, quickstart, commands, outputs, docs, and limitations." \
  "README.md" \
  "- quickstart runs locally\n- examples are honest\n- command list matches Makefile"

issue "Generate resume bullets, LinkedIn post, and portfolio summary" \
  "Milestone 6: Documentation and Portfolio Polish" \
  "portfolio,docs" \
  "Prepare public portfolio materials for LLM/AI systems roles." \
  "docs/resume_bullets.md, docs/linkedin_post.md, docs/portfolio_summary.md" \
  "- materials are technical and non-cringey\n- bullets target multiple role families\n- scale claims are honest"

issue "Polish repo for public portfolio launch" \
  "Milestone 6: Documentation and Portfolio Polish" \
  "portfolio,devops,docs" \
  "Finalize ignored artifacts, validation checks, GitHub metadata, and launch instructions." \
  ".gitignore, README.md, docs/github_issues_plan.md" \
  "- no large generated artifacts are staged\n- validation commands pass\n- GitHub issue plan is ready to run"
```

## Current Completion Mapping

Most implementation issues above are complete in the initial codebase. Keep the
issues open if you want them to preserve the professional project history, then
close them as you verify each milestone on GitHub.
