"""The end-to-end RAG pipeline: ingest, retrieve, rerank, and answer with citations.

Design goals (an enterprise/scientific assistant, not a notebook demo):

* **Reproducible indexing** — deterministic chunk ids and embeddings.
* **Provenance** — every chunk carries its source path and character offsets.
* **Grounded answers** — the generator is constrained to the retrieved context.
* **Honest refusals** — when no chunk clears the score threshold, the assistant
  refuses instead of hallucinating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dork.generation.providers import LanguageModel, build_language_model
from dork.rag.chunking import chunk_text
from dork.rag.embeddings import EmbeddingBackend, build_embedder
from dork.rag.loaders import load_documents
from dork.rag.reranker import LexicalReranker
from dork.rag.schema import Chunk, Citation, Document, RagAnswer, ScoredChunk
from dork.rag.vectorstore import VectorStore, build_vector_store
from dork.utils.config import RagConfig
from dork.utils.logging import get_logger

logger = get_logger(__name__)

REFUSAL_TEXT = (
    "I don't have enough information in the provided documents to answer that confidently."
)


@dataclass
class IngestStats:
    """Summary of an ingestion run."""

    documents: int
    chunks: int
    embedding_dim: int
    store_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "embedding_dim": self.embedding_dim,
            "store_count": self.store_count,
        }


class RagPipeline:
    """A configurable retrieval-augmented generation pipeline."""

    def __init__(
        self,
        cfg: RagConfig,
        embedder: EmbeddingBackend | None = None,
        store: VectorStore | None = None,
        model: LanguageModel | None = None,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder or build_embedder(cfg.embeddings)
        self.store = store or build_vector_store(cfg.vector_store)
        self.model = model or build_language_model(cfg.generation)
        self.reranker = LexicalReranker()

    # ── Ingestion ────────────────────────────────────────────────────
    def ingest(self, source_dir: str | None = None) -> IngestStats:
        """Load, chunk, embed and index documents from ``source_dir``."""
        ing = self.cfg.ingest or {}
        source = source_dir or ing.get("source_dir", "data/sample_docs")
        docs = load_documents(source, ing.get("glob", "**/*"))
        chunks = self._chunk_documents(docs)
        if not chunks:
            logger.warning("No chunks produced from %s", source)
            return IngestStats(len(docs), 0, self.embedder.dim, self.store.count())

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        self.store.persist()
        stats = IngestStats(len(docs), len(chunks), self.embedder.dim, self.store.count())
        logger.info("Ingested %s", stats.to_dict())
        return stats

    def _chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        ck = (self.cfg.ingest or {}).get("chunking", {})
        chunks: list[Chunk] = []
        for doc in docs:
            spans = chunk_text(
                doc.text,
                strategy=ck.get("strategy", "recursive"),
                chunk_size=int(ck.get("chunk_size", 512)),
                chunk_overlap=int(ck.get("chunk_overlap", 64)),
                min_chunk_chars=int(ck.get("min_chunk_chars", 64)),
            )
            for i, span in enumerate(spans):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{i}",
                        doc_id=doc.doc_id,
                        source=doc.source,
                        text=span.text,
                        index=i,
                        start=span.start,
                        end=span.end,
                        metadata=dict(doc.metadata),
                    )
                )
        return chunks

    # ── Retrieval ────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Embed the query, fetch top-k chunks, optionally rerank and filter."""
        ret = self.cfg.retrieval or {}
        top_k = top_k or int(ret.get("top_k", 5))
        q_emb = self.embedder.embed_one(query)
        hits = self.store.search(q_emb, top_k)

        if ret.get("rerank", True) and hits:
            hits = self.reranker.rerank(query, hits, int(ret.get("rerank_top_n", 3)))

        min_score = float(ret.get("min_score", 0.0))
        return [h for h in hits if h.score >= min_score]

    # ── Generation ───────────────────────────────────────────────────
    def query(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Answer ``question`` grounded in retrieved evidence, with citations."""
        gen = self.cfg.generation or {}
        contexts = self.retrieve(question, top_k)

        if not contexts and gen.get("refuse_when_insufficient", True):
            return RagAnswer(question, REFUSAL_TEXT, refused=True, model=self.model.name)

        prompt = self._build_prompt(question, contexts)
        raw = self.model.complete(
            prompt,
            max_new_tokens=int(gen.get("max_new_tokens", 256)),
            temperature=float(gen.get("temperature", 0.0)),
        )
        citations = self._extract_citations(raw, contexts)
        refused = any(k in raw.lower() for k in ("don't have enough", "insufficient"))
        return RagAnswer(
            question=question,
            answer=raw.strip(),
            citations=citations,
            contexts=contexts,
            refused=refused,
            model=self.model.name,
        )

    def _build_prompt(self, question: str, contexts: list[ScoredChunk]) -> str:
        ctx_block = "\n".join(f"[{i + 1}] {sc.chunk.text}" for i, sc in enumerate(contexts))
        return (
            "You are a precise research assistant. Answer the question using ONLY the "
            "context below. Cite supporting sources inline as [n]. If the context is "
            "insufficient, reply that you don't have enough information.\n\n"
            f"Context:\n{ctx_block}\n\nQuestion: {question}\nAnswer:"
        )

    @staticmethod
    def _extract_citations(answer: str, contexts: list[ScoredChunk]) -> list[Citation]:
        from dork.evaluation.metrics import extract_citations

        markers = sorted(set(extract_citations(answer)))
        citations: list[Citation] = []
        for m in markers:
            if 1 <= m <= len(contexts):
                sc = contexts[m - 1]
                citations.append(
                    Citation(
                        marker=m,
                        source=sc.chunk.source,
                        chunk_id=sc.chunk.chunk_id,
                        snippet=sc.chunk.text[:200],
                        score=sc.score,
                    )
                )
        return citations


def build_pipeline(cfg: RagConfig) -> RagPipeline:
    """Factory mirroring the other subsystems' ``build_*`` helpers."""
    return RagPipeline(cfg)
