"""Corpus preparation for the tiny GPT.

Only small *public* datasets are used. Each loader is local-first: it attempts a
download when needed, but always falls back to a bundled public-domain sample so
training, tests and CI work fully offline. No proprietary or sensitive data.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from dork.utils.config import DataConfig
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)

# Public-domain (Shakespeare, ~16th c.) text. Bundled so the project trains and
# self-tests with zero network access; the real corpora are larger downloads.
FALLBACK_CORPUS = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die-to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream-ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them,
The good is oft interred with their bones;
So let it be with Caesar.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

If music be the food of love, play on;
Give me excess of it, that, surfeiting,
The appetite may sicken, and so die.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest:
It blesseth him that gives and him that takes.

We are such stuff as dreams are made on,
And our little life is rounded with a sleep.

Once more unto the breach, dear friends, once more;
Or close the wall up with our English dead!
In peace there's nothing so becomes a man
As modest stillness and humility.

Good night, good night! parting is such sweet sorrow,
That I shall say good night till it be morrow.
"""

DATASET_URLS = {
    "tiny_shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    ),
}


def _download(url: str, dest: Path, timeout: float = 20.0) -> bool:
    """Best-effort download. Returns True on success, False on any failure."""
    try:
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        logger.info("Saved %d bytes to %s", len(data), dest)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Download failed (%s); using bundled fallback corpus.", exc)
        return False


def prepare_corpus(cfg: DataConfig) -> Path:
    """Ensure the raw training corpus exists on disk and return its path.

    Resolution order:
        1. If ``raw_path`` already exists, reuse it (reproducible, cached).
        2. For known datasets, attempt a download.
        3. On any failure, write the bundled public-domain fallback corpus.

    Args:
        cfg: The validated :class:`~dork.utils.config.DataConfig`.

    Returns:
        Absolute path to the prepared UTF-8 text corpus.
    """
    raw_path = resolve_path(cfg.raw_path, create_parent=True)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        logger.info("Using cached corpus at %s (%d bytes)", raw_path, raw_path.stat().st_size)
        return raw_path

    if cfg.dataset == "custom":
        raise FileNotFoundError(
            f"dataset=custom requires an existing file at {raw_path}. "
            "Point data.raw_path at a local UTF-8 text file."
        )

    if cfg.dataset == "tiny_shakespeare" and _download(DATASET_URLS["tiny_shakespeare"], raw_path):
        return raw_path

    if cfg.dataset in {"tinystories", "wikitext2"}:
        text = _try_hf_dataset(cfg.dataset)
        if text:
            raw_path.write_text(text, encoding="utf-8")
            return raw_path

    # Universal offline fallback.
    logger.info("Writing bundled fallback corpus to %s", raw_path)
    raw_path.write_text(FALLBACK_CORPUS, encoding="utf-8")
    return raw_path


def _try_hf_dataset(name: str, max_chars: int = 2_000_000) -> str | None:
    """Try to materialize a HF dataset to plain text; return None if unavailable."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        logger.warning("`datasets` not installed; cannot fetch %s.", name)
        return None

    try:
        if name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            key = "text"
        else:  # wikitext2
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            key = "text"

        parts: list[str] = []
        total = 0
        for row in ds:
            txt = (row.get(key) or "").strip()
            if not txt:
                continue
            parts.append(txt)
            total += len(txt)
            if total >= max_chars:
                break
        return "\n".join(parts) if parts else None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to load %s from HF (%s).", name, exc)
        return None
