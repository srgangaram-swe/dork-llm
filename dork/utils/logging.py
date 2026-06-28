"""Structured, colorized logging built on :mod:`rich` with a stdlib fallback.

A single :func:`get_logger` is used across the codebase so every subsystem logs
with a consistent format and respects the ``DORK_LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single handler to the ``dork`` root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.environ.get("DORK_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("dork")
    logger.setLevel(level)
    logger.propagate = False

    try:  # Prefer rich for readable, colorized output when available.
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            log_time_format="%H:%M:%S",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:  # pragma: no cover - exercised only without rich
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )

    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger under the ``dork`` root.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    _configure_root()
    if not name.startswith("dork"):
        name = f"dork.{name}"
    return logging.getLogger(name)
