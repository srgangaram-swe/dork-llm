"""Tests for the char tokenizer (and BPE when the extra is installed)."""

from __future__ import annotations

import pytest
from dork.tokenizer.char import CharTokenizer


def test_char_roundtrip(sample_text):
    tok = CharTokenizer.train(sample_text)
    ids = tok.encode(sample_text)
    assert tok.decode(ids) == sample_text  # specials excluded, but none injected
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_char_special_tokens_reserved():
    tok = CharTokenizer.train("abc", ["<|endoftext|>", "<|pad|>", "<|unk|>"])
    assert tok.stoi["<|endoftext|>"] == 0
    assert tok.vocab_size == 3 + len(set("abc"))


def test_char_unknown_maps_to_unk():
    tok = CharTokenizer.train("abc")
    # 'z' is unseen -> maps to <|unk|> id, which decode drops.
    ids = tok.encode("z")
    assert ids == [tok.unk_id]


def test_char_save_load(tmp_path, sample_text):
    tok = CharTokenizer.train(sample_text)
    p = tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(p)
    assert loaded.encode(sample_text) == tok.encode(sample_text)


def test_bpe_if_available(sample_text):
    pytest.importorskip("tokenizers")
    from dork.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.train(sample_text, vocab_size=300)
    ids = tok.encode("To be")
    assert isinstance(ids, list) and len(ids) > 0
    assert "To be" in tok.decode(ids)
