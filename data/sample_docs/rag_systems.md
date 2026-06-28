# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) grounds a language model's answers in
external evidence. Instead of relying only on parametric memory, the system
retrieves relevant passages and conditions generation on them.

## Indexing pipeline

Documents are chunked into smaller passages, each passage is embedded into a
dense vector, and the vectors are stored in a vector database. Each chunk keeps
metadata such as its source path, chunk index, and character offsets, which makes
precise citations possible.

## Retrieval and reranking

At query time the question is embedded and the most similar chunks are retrieved
by cosine similarity. A reranking stage can reorder the retrieved chunks by
relevance to improve precision. The top reranked chunks are passed to the
generator as context.

## Grounded answers and citations

Grounded answers cite the retrieved chunks that support each claim, usually with
inline markers such as [1] or [2]. This lets a reader verify the answer against
its sources. Citation coverage measures how often answers include valid
citations, and faithfulness measures whether the answer is actually supported by
the cited context.

## Refusing when evidence is insufficient

A trustworthy assistant refuses to answer when retrieval finds no chunk above the
score threshold. Refusing avoids hallucination, which is when a model produces
fluent but unsupported claims.
