#!/usr/bin/env python3
"""Write the shareable surrogate dataset (data/surrogate/) from local real data.

The surrogate keeps the real numeric game stats but drops every name and
piece of descriptive text (monster names, type lines, damage dice strings,
parser flags). Each record gets an opaque id instead. Records keep their
original order so that seeded train/calibration/test splits, and therefore
every fitted model, are identical whether fit on real or surrogate data.

Only fields on the allowlists below are written; anything new the parsers
add stays local until it is deliberately added here.

Usage:
    .venv/bin/python -m pipeline.make_surrogate
"""

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.datasets import (  # noqa: E402
    SURROGATE_DIR, SURROGATE_PAIRS, SURROGATE_SIDES, load_pairs, load_sides,
)

ABILITIES = [f"ability_{a}" for a in ("str", "dex", "con", "int", "wis", "cha")]
SAVES = [f"save_{a}" for a in ("str", "dex", "con", "int", "wis", "cha")]

# role (e.g. "Elite Soldier") is generic 4e mechanical vocabulary, kept
# because it drives 4e HP/damage math.
KEEP = {
    "4e": ["level", "role", "xp", "hp", "ac", "fort", "ref", "will",
           "attack_bonus", "damage_avg", *ABILITIES],
    "5e": ["cr_num", "xp", "hp", "ac", "attack_bonus", "damage_avg",
           *ABILITIES, *SAVES],
}


def mask(side, edition, record_id):
    out = {"id": record_id}
    for key in KEEP[edition]:
        if side.get(key) is not None:
            out[key] = side[key]
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    SURROGATE_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs("real")
    masked_pairs = [
        {
            "id": f"pair-{i:04d}",
            "4e": mask(p["4e"], "4e", f"pair-{i:04d}/4e"),
            "5e": mask(p["5e"], "5e", f"pair-{i:04d}/5e"),
        }
        for i, p in enumerate(pairs, start=1)
    ]
    write_jsonl(SURROGATE_PAIRS, masked_pairs)
    print(f"Wrote {len(masked_pairs)} pairs -> {SURROGATE_PAIRS.relative_to(SURROGATE_DIR.parent.parent)}")

    for edition, path in SURROGATE_SIDES.items():
        sides = load_sides(edition, "real")
        rows = [mask(s, edition, f"m{edition}-{i:04d}") for i, s in enumerate(sides, start=1)]
        write_jsonl(path, rows)
        print(f"Wrote {len(rows)} {edition} monsters -> {path.relative_to(SURROGATE_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
