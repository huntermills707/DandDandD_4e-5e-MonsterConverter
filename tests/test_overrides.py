"""Override mechanics, exercised with the invented example files."""

import copy
import json

from parsers.overrides import apply_overrides
from pipeline.datasets import ROOT

EXAMPLE_4E = ROOT / "manual" / "overrides_4e.example.json"
EXAMPLE_5E = ROOT / "manual" / "overrides_5e.example.json"


def test_example_files_are_valid():
    for path in (EXAMPLE_4E, EXAMPLE_5E):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["overrides"]
        for ov in data["overrides"]:
            assert ov["match"]
            assert ("set" in ov) != ov.get("drop", False)


def test_4e_patch_by_name():
    monsters = [
        {"name": "Example Bog Lurker", "level": 4, "abilities": {}},
        {"name": "Example Bog Lurker", "level": 9, "abilities": {}},
        {"name": "Example Cinder Hound", "level": 7, "attacks": []},
    ]
    assert apply_overrides(monsters, EXAMPLE_4E) == 2
    assert monsters[0]["abilities"]["dex"] == 16
    assert monsters[0]["manual_fix"]
    assert monsters[1]["abilities"] == {}  # level must match too
    assert monsters[2]["attacks"][0]["bonus"] == 12


def test_5e_rename_by_stats_and_drop():
    monsters = [
        {"cr": "3", "hp": 58, "type_line": "Medium Construct, unaligned"},
        {"cr": "3", "hp": 58, "type_line": "Medium beast, unaligned"},
        {"cr": "1/2", "hp": 19, "name": "Garbled"},
    ]
    assert apply_overrides(monsters, EXAMPLE_5E) == 2
    assert monsters[0]["name"] == "Example Brass Sentinel"
    assert "name" not in monsters[1]
    assert len(monsters) == 2  # dropped block removed


def test_ambiguous_match_is_skipped():
    monsters = [{"cr": "1/2", "hp": 19}, {"cr": "1/2", "hp": 19}]
    before = copy.deepcopy(monsters)
    assert apply_overrides(monsters, EXAMPLE_5E) == 0
    assert monsters == before
