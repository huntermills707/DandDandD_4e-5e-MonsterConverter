"""The tracked surrogate must carry numbers only: no names or book text."""

import json
import re

import pytest

from pipeline.datasets import SURROGATE_PAIRS, SURROGATE_SIDES
from pipeline.make_surrogate import KEEP

ROLE_WORDS = {"Minion", "Standard", "Elite", "Solo", "Brute", "Soldier",
              "Artillery", "Lurker", "Skirmisher", "Controller", "Leader"}
FORBIDDEN_KEYS = {"name", "name_caps", "type_line", "damage_dice", "cr",
                  "text", "attacks", "languages", "senses", "skills", "flags"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_side(side, edition):
    assert set(side) - {"id"} <= set(KEEP[edition]), set(side) - set(KEEP[edition])
    assert not set(side) & FORBIDDEN_KEYS
    for key, value in side.items():
        if key == "id":
            assert re.fullmatch(r"(pair-\d{4}/(4e|5e))|(m(4e|5e)-\d{4})", value)
        elif key == "role":
            words = set(value.replace("(", " ").replace(")", " ").split())
            assert words <= ROLE_WORDS, value
        else:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (key, value)


def test_pairs_are_masked():
    pairs = read(SURROGATE_PAIRS)
    assert len(pairs) > 100
    for p in pairs:
        assert set(p) == {"id", "reason", "match_type", "group", "4e", "5e"}
        assert isinstance(p["group"], int)
        assert p["reason"] in {"exact", "alias", "ocr", "model", "fuzzy"}
        assert p["match_type"] in {"same", "related"}
        check_side(p["4e"], "4e")
        check_side(p["5e"], "5e")


@pytest.mark.parametrize("edition", ["4e", "5e"])
def test_sides_are_masked(edition):
    rows = read(SURROGATE_SIDES[edition])
    assert len(rows) > 300
    for row in rows:
        check_side(row, edition)
