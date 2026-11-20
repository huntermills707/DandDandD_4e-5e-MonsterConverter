"""Cardinality modes on synthetic pairs (5e ids only, no names)."""

import pytest

from pipeline.cardinality import select


def pair(i5, score, same=False):
    return {"i5": i5, "score": score, "match_type": "same" if same else "related"}


FLAT = [pair(1, s) for s in (0.036, 0.035, 0.035, 0.034)]             # one family, no elbow
CLIFF = [pair(2, 0.28), pair(2, 0.027), pair(2, 0.026)]                # lone standout
ANCHORED = [pair(3, 0.13, same=True), pair(3, 0.26), pair(3, 0.01)]    # exact anchor not top
TWO_ANCHORS = [pair(4, 0.6, same=True), pair(4, 0.2, same=True), pair(4, 0.19)]


def kept_scores(pairs, mode):
    kept, dropped = select(pairs, mode)
    assert len(kept) + len(dropped) == len(pairs)
    return sorted((p["score"] for p in kept), reverse=True)


def test_many_keeps_everything():
    assert len(select(FLAT + CLIFF, "many")[0]) == 7


def test_one_prefers_the_exact_anchor():
    assert kept_scores(ANCHORED, "one") == [0.13]
    assert kept_scores(CLIFF, "one") == [0.28]


def test_elbow():
    assert kept_scores(FLAT, "elbow") == [0.036, 0.035, 0.035, 0.034]
    assert kept_scores(CLIFF, "elbow") == [0.28]
    assert kept_scores(ANCHORED, "elbow") == [0.26, 0.13]          # anchor, then 0.26; 0.01 cut
    assert kept_scores(TWO_ANCHORS, "elbow") == [0.6, 0.2, 0.19]   # all anchors kept


def test_unknown_mode():
    with pytest.raises(ValueError):
        select(FLAT, "some")
