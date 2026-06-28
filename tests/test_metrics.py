"""Tests for evaluation metric helpers."""

from __future__ import annotations

from dork.evaluation import metrics as M


def test_normalize_and_exact_match():
    assert M.exact_match("The Answer.", "answer")
    assert not M.exact_match("no", "yes")


def test_contains_answer():
    assert M.contains_answer("The result is 42 indeed", "42")
    assert not M.contains_answer("nothing here", "42")


def test_token_f1():
    assert M.token_f1("a b c", "a b c") == 1.0
    assert M.token_f1("a b", "c d") == 0.0
    assert 0 < M.token_f1("a b c", "a b") < 1


def test_extract_json_plain_and_fenced():
    assert M.is_valid_json('{"a": 1}')[0]
    assert M.is_valid_json('```json\n{"a": 1}\n```')[0]
    assert not M.is_valid_json("not json")[0]


def test_schema_and_coverage():
    obj = {"name": "x", "age": 1}
    assert M.schema_ok(obj, ["name", "age"])
    assert not M.schema_ok(obj, ["name", "missing"])
    assert M.key_coverage(obj, ["name", "missing"]) == 0.5


def test_choice_and_citation_extraction():
    assert M.extract_choice_letter("The answer is C.") == "C"
    assert M.extract_citations("Grounded [1] and also [3].") == [1, 3]
