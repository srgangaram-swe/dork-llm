# Data

Dork LLM is designed to be clean-room and public-data only.

Tracked data:

- `sample_docs/`: small Markdown documents used by the RAG and agent demos.
- `README.md`: this policy and regeneration guide.

Ignored generated data:

- `data/raw/`: downloaded public corpora such as Tiny Shakespeare.
- `data/processed/` and `*.bin`: tokenized train/validation files.
- vector stores, checkpoints, tokenizers, reports, and other local artifacts.

The default training config uses Tiny Shakespeare from Andrej Karpathy's public
char-rnn sample corpus when network access is available. If the download fails,
the loader writes a bundled public-domain Shakespeare excerpt so tests and smoke
runs remain offline and reproducible.

Regenerate local data with:

```bash
make prepare-data
make train-tokenizer
make train-small-gpt
make ingest-docs
make eval
```

Do not add employer data, private documents, CUI, secrets, production logs, or
large third-party datasets to this repository.
