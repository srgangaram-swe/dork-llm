"""Vector stores behind a common interface.

* :class:`MemoryVectorStore` — a NumPy cosine-similarity index with JSON
  persistence. Zero external dependencies; ideal for local-first use and tests.
* :class:`ChromaVectorStore` — a thin wrapper over ChromaDB for a persistent,
  production-style store.
"""

from __future__ import annotations

import abc

import numpy as np

from dork.rag.schema import Chunk, ScoredChunk
from dork.utils.io import load_json, save_json
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)


class VectorStore(abc.ABC):
    """Add embedded chunks and query by vector similarity."""

    @abc.abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abc.abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    def persist(self) -> None:  # noqa: B027  # optional no-op hook
        """Persist the index if the backend supports it (no-op by default)."""

    def reset(self) -> None:  # noqa: B027  # optional no-op hook
        """Clear all stored vectors (no-op by default)."""


class MemoryVectorStore(VectorStore):
    """In-memory cosine index backed by a single NumPy matrix."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self.persist_dir = persist_dir

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        emb = _l2_normalize(embeddings.astype(np.float32))
        self._matrix = emb if self._matrix is None else np.vstack([self._matrix, emb])
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []
        q = _l2_normalize(query_embedding.reshape(1, -1).astype(np.float32))[0]
        sims = self._matrix @ q  # cosine since both are L2-normalized
        k = min(top_k, len(self._chunks))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [ScoredChunk(self._chunks[i], float(sims[i])) for i in top_idx]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks = []
        self._matrix = None

    def persist(self) -> None:
        if not self.persist_dir or self._matrix is None:
            return
        out = resolve_path(self.persist_dir, create_parent=True)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self._matrix)
        save_json(out / "chunks.json", [c.to_dict() for c in self._chunks])
        logger.info("Persisted %d vectors to %s", self.count(), out)

    @classmethod
    def load(cls, persist_dir: str) -> MemoryVectorStore:
        out = resolve_path(persist_dir)
        store = cls(persist_dir=persist_dir)
        emb_path = out / "embeddings.npy"
        chunk_path = out / "chunks.json"
        if emb_path.exists() and chunk_path.exists():
            store._matrix = np.load(emb_path)
            store._chunks = [Chunk(**_drop_score(d)) for d in load_json(chunk_path)]
            logger.info("Loaded %d vectors from %s", store.count(), out)
        return store


class ChromaVectorStore(VectorStore):
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: str = ".chroma", collection: str = "dork_docs") -> None:
        import chromadb  # type: ignore

        path = str(resolve_path(persist_dir, create_parent=True))
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings.astype(np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                }
                for c in chunks
            ],
        )

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        res = self._col.query(
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            n_results=top_k,
        )
        out: list[ScoredChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            chunk = Chunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                text=doc,
                index=int(meta.get("index", 0)),
                start=int(meta.get("start", 0)),
                end=int(meta.get("end", 0)),
            )
            out.append(ScoredChunk(chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(self._col.name)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _drop_score(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "score"}


def build_vector_store(cfg: dict) -> VectorStore:
    """Construct a vector store from config; fall back to memory offline."""
    backend = str(cfg.get("backend", "memory")).lower()
    persist_dir = cfg.get("persist_dir", ".chroma")
    if backend == "chroma":
        try:
            return ChromaVectorStore(persist_dir, cfg.get("collection", "dork_docs"))
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s); using MemoryVectorStore.", exc)
    return MemoryVectorStore(persist_dir=persist_dir)
