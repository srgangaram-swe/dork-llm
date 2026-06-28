"""Path helpers that keep the project free of hardcoded absolute paths.

Everything is resolved relative to the repository root (the directory containing
``pyproject.toml``) so the code behaves identically regardless of the working
directory it is launched from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root by walking up to find ``pyproject.toml``.

    Falls back to the current working directory if no marker is found (e.g. when
    the package is installed as a wheel rather than run from a checkout).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def resolve_path(path: str | Path, *, create_parent: bool = False) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute.

    Args:
        path: A relative or absolute path.
        create_parent: If True, ensure the parent directory exists.

    Returns:
        An absolute :class:`~pathlib.Path`.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if create_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
