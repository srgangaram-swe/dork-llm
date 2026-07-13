# RAG Design

## Goal

AxiomStack includes a local-first retrieval-augmented generation system that can
ingest documents, retrieve evidence, answer with citations, and refuse when
evidence is insufficient.

The system is intentionally lightweight, but the architecture mirrors production
RAG services:

```text
documents -> loaders -> chunking -> embeddings -> vector store
          -> retrieve -> rerank -> grounded prompt -> answer + citations
```

## Entry Points

```bash
make ingest-docs
make query-rag Q="What does causal masking prevent?"
dork ingest --source data/sample_docs
dork query --question "What does causal masking prevent?"
```

Configuration: `configs/rag_default.yaml`.

## Ingestion

`dork.rag.loaders` supports:

- Markdown
- plain text
- PDFs when `pypdf` is installed

Each loaded document becomes a `Document` with text, source path, doc id, and
metadata. The chunker then produces spans with character offsets so citations
can point back to exact source text.

Chunking strategies:

- `recursive`: paragraph-aware fallback to word chunks.
- `fixed`: deterministic fixed-size windows.
- `sentence`: sentence-oriented chunks.

## Embeddings

Two backends are supported:

- `hash`: deterministic hashing embedder with no network or model download.
- `sentence_transformers`: small open embedding model such as
  `sentence-transformers/all-MiniLM-L6-v2`.

The hash embedder is useful for CI and demonstrations. Sentence-transformers is
better for semantic retrieval quality.

## Vector Stores

Two stores are supported:

- `memory`: NumPy-backed in-memory search with optional local persistence.
- `chroma`: ChromaDB-backed persistence when installed.

Generated stores are ignored by git. Rebuild them with `make ingest-docs`.

## Retrieval and Reranking

The pipeline embeds the query, retrieves top-k chunks, optionally reranks them
with a lexical overlap reranker, filters by `min_score`, and passes the surviving
contexts into the generator.

If no chunk clears the threshold and `refuse_when_insufficient` is true, the
assistant returns:

```text
I don't have enough information in the provided documents to answer that confidently.
```

## Citations

The grounded prompt numbers each retrieved chunk as `[1]`, `[2]`, and so on. The
answer parser extracts citation markers and maps them back to:

- source file;
- chunk id;
- snippet;
- retrieval score.

This makes the final `RagAnswer` auditable and usable by DorkChat, the API, and
the dashboard.

## Evaluation

`rag_faithfulness` checks:

- valid citation coverage;
- token overlap between answer and cited evidence;
- refusal behavior for unanswerable synthetic prompts.

This is a lightweight regression signal. Stronger future work could add LLM-as-
judge entailment, claim decomposition, and citation-level precision/recall.
