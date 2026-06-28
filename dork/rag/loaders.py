"""Document loaders for plain text, Markdown and PDF sources.

PDF parsing uses ``pypdf`` when available; if it is missing, PDFs are skipped
with a warning rather than crashing the ingestion run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dork.rag.schema import Document
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def load_document(path: str | Path) -> Document | None:
    """Load a single document; return None for unsupported/empty files."""
    p = resolve_path(path)
    ext = p.suffix.lower()
    if ext in TEXT_EXTS:
        text = p.read_text(encoding="utf-8", errors="ignore")
    elif ext in PDF_EXTS:
        text = _read_pdf(p)
    else:
        return None
    text = text.strip()
    if not text:
        return None
    rel = _relative_source(p)
    return Document(
        doc_id=_doc_id(rel),
        source=rel,
        text=text,
        metadata={"ext": ext, "chars": len(text)},
    )


def load_documents(source_dir: str | Path, glob: str = "**/*") -> list[Document]:
    """Recursively load all supported documents under ``source_dir``."""
    root = resolve_path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.glob(glob)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            doc = load_document(path)
            if doc:
                docs.append(doc)
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF %s", path.name)
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # pragma: no cover - depends on file
        logger.warning("Failed to parse PDF %s (%s)", path.name, exc)
        return ""


def _relative_source(p: Path) -> str:
    """Return a repo-relative source path when possible (clean citations)."""
    from dork.utils.paths import project_root

    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return p.name
