"""Load modeling inputs from either the real local data or the tracked surrogate.

Real data (data/raw, data/processed) is derived from the rule book PDFs and
never leaves this machine. The surrogate (data/surrogate) holds the same
numeric stats in the same order, with names and descriptive text removed.
Model fits therefore come out identical on either source; see
pipeline/make_surrogate.py and tests/test_reproduce_models.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
SURROGATE_DIR = ROOT / "data" / "surrogate"

REAL_PAIRS = PROC_DIR / "training_pairs.jsonl"
REAL_RAW = {"4e": RAW_DIR / "mm4_monsters.jsonl", "5e": RAW_DIR / "mm5_monsters.jsonl"}

SURROGATE_PAIRS = SURROGATE_DIR / "training_pairs.jsonl"
SURROGATE_SIDES = {
    "4e": SURROGATE_DIR / "monsters_4e.jsonl",
    "5e": SURROGATE_DIR / "monsters_5e.jsonl",
}

SOURCES = ("auto", "real", "surrogate")


def resolve_source(source="auto"):
    """'auto' picks the real data when it exists locally, else the surrogate."""
    if source not in SOURCES:
        raise ValueError(f"unknown data source {source!r}; expected one of {SOURCES}")
    if source != "auto":
        return source
    real_ok = REAL_PAIRS.exists() and all(p.exists() for p in REAL_RAW.values())
    return "real" if real_ok else "surrogate"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(source="auto"):
    """High-confidence matched 4e/5e pairs: [{'4e': {...}, '5e': {...}}, ...]."""
    source = resolve_source(source)
    return _read_jsonl(REAL_PAIRS if source == "real" else SURROGATE_PAIRS)


def load_sides(edition, source="auto"):
    """Flat per-edition stat records, unresolved OCR suspects removed."""
    source = resolve_source(source)
    if source == "surrogate":
        return _read_jsonl(SURROGATE_SIDES[edition])

    from pipeline.build_pairs import build_side

    sides = [build_side(r, edition) for r in _read_jsonl(REAL_RAW[edition])]
    return [
        s for s in sides
        if not (s["flags"]["ocr_suspect"] and not s["flags"]["manual_fix"])
    ]
