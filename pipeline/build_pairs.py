#!/usr/bin/env python3
"""Build cross-edition matched pairs from parsed 4e and 5e stat blocks.

Outputs:
    data/processed/matched_pairs.csv   - every candidate pair with a score and
                                         a review flag.
    data/processed/training_pairs.jsonl - high-confidence pairs (score >= 0.85,
                                          no unresolved OCR flags) ready for
                                          regression modeling.

The matcher uses stdlib only (json, csv, difflib, re).  Names are normalized,
edition-specific dragon age synonyms are aligned (4e "Elder" == 5e "Ancient"),
parenthetical edition tags are stripped, and fuzzy matching is gated by token
overlap and age rules so that different colors/ages of dragon or different
species of "giant X" are not forced into pairs.
"""

import csv
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

MM4_PATH = RAW_DIR / "mm4_monsters.jsonl"
MM5_PATH = RAW_DIR / "mm5_monsters.jsonl"
PAIRS_CSV = PROC_DIR / "matched_pairs.csv"
TRAINING_PATH = PROC_DIR / "training_pairs.jsonl"
SRD_PATH = PROC_DIR / "srd_5e.json"
SRD_URL = "https://api.open5e.com/v1/monsters/?limit=1000&document__slug=wotc-srd"

# Bases that are too generic to promote via substring matching.
VARIANT_PROMOTE_BLOCKLIST = {"shadow"}

# Age/size words that are aligned across editions for matching.
AGE_SYNONYMS = {"elder": "ancient"}  # 4e Elder Dragon -> 5e Ancient Dragon

# Stopwords removed when building the semantic "core" tokens used to decide
# whether a fuzzy candidate is plausible.  This lets "Giant Rat" match
# "Diseased Giant Rat" while preventing "Giant Rat" from matching "Giant Goat".
STOPWORDS = {
    "the", "a", "an", "of", "and",
    "young", "adult", "ancient", "elder", "wyrmling", "juvenile",
    "giant", "swarm", "large", "medium", "small", "tiny", "huge", "gargantuan",
    "dragon", "dracolich",
}

# Age tokens that must agree when both names have them.
AGE_TOKENS = {"young", "adult", "ancient", "elder", "wyrmling", "juvenile"}


def normalize(name: str) -> str:
    """Lowercase, drop parentheticals, drop punctuation/digits, collapse ws."""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", "", s)
    return " ".join(s.split())


def edition_normalize(name: str) -> str:
    """Cross-edition normalization: Elder -> Ancient, etc."""
    tokens = normalize(name).split()
    return " ".join(AGE_SYNONYMS.get(t, t) for t in tokens)


def core_tokens(name: str) -> set:
    """Semantic tokens after stripping common modifiers/parentheticals."""
    return set(edition_normalize(name).split()) - STOPWORDS


def age_canon(name: str) -> str:
    """Return the canonical age word in the name, if any."""
    tokens = set(normalize(name).split()) & AGE_TOKENS
    if not tokens:
        return None
    age = sorted(tokens)[0]
    return AGE_SYNONYMS.get(age, age)


def load_srd_reference():
    """Load a local 5e SRD cache; fetch from Open5e if missing."""
    if not SRD_PATH.exists():
        try:
            import urllib.request
            print(f"Fetching SRD reference from {SRD_URL} ...")
            urllib.request.urlretrieve(SRD_URL, SRD_PATH)
        except Exception as e:
            print(f"Could not fetch SRD reference: {e}")
            return {}
    try:
        with open(SRD_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {r["name"]: r for r in data.get("results", [])}
    except Exception as e:
        print(f"Could not load SRD reference: {e}")
        return {}


def srd_verified(name: str, cr, hp: int, ref: dict) -> bool:
    """True if the parsed stats match the SRD entry for this name."""
    if not ref or name not in ref:
        return False
    srd = ref[name]
    return str(cr) == str(srd.get("challenge_rating")) and int(hp) == srd.get("hit_points")


def is_variant_promotion(name4: str, name5: str, score: float) -> bool:
    """Promote obvious 4e variant -> 5e base pairs (e.g., Fire Titan -> Fire Giant)."""
    if score < 0.65:
        return False
    n4 = normalize(name4)
    n5 = normalize(name5)
    if n5 in VARIANT_PROMOTE_BLOCKLIST:
        return False
    # Require the 4e name to start with the 5e base name; this avoids false
    # matches like "Death Knight" -> "Knight" while catching real variants.
    return n4.startswith(n5)


def parse_cr(cr) -> float:
    if cr is None:
        return None
    s = str(cr).strip()
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            return int(num) / int(den)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def avg_damage(dice: str) -> float:
    """Average of a dice expression like '2d8+8' or '1d6'."""
    if not dice:
        return None
    dice = dice.lower().replace(" ", "").replace("−", "-")
    m = re.match(r"^(?:(\d+)d(\d+))?(?:\+?(-?\d+))?$", dice)
    if not m:
        try:
            return float(dice)
        except ValueError:
            return None
    count = int(m.group(1) or 0)
    sides = int(m.group(2) or 0)
    mod = int(m.group(3) or 0)
    return count * (sides + 1) / 2 + mod


def pick_attack(attacks):
    """Choose the attack with the highest average damage."""
    if not attacks:
        return None
    scored = []
    for a in attacks:
        dmg = None
        if "damage_avg" in a and a["damage_avg"] is not None:
            dmg = float(a["damage_avg"])
        elif "damage_dice" in a and a["damage_dice"]:
            dmg = avg_damage(a["damage_dice"])
        bonus = a.get("bonus")
        scored.append((dmg if dmg is not None else -1, bonus, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][2]


def attack_summary(attacks):
    a = pick_attack(attacks)
    if not a:
        return {}
    dmg = None
    if "damage_avg" in a and a["damage_avg"] is not None:
        dmg = float(a["damage_avg"])
    elif "damage_dice" in a and a["damage_dice"]:
        dmg = avg_damage(a["damage_dice"])
    return {
        "attack_bonus": a.get("bonus"),
        "damage_avg": dmg,
        "damage_dice": a.get("damage_dice"),
    }


def abilities(rec):
    ab = rec.get("abilities") or {}
    return {
        "str": ab.get("str"),
        "dex": ab.get("dex"),
        "con": ab.get("con"),
        "int": ab.get("int"),
        "wis": ab.get("wis"),
        "cha": ab.get("cha"),
    }


def build_side(rec, edition: str):
    ab = abilities(rec)
    atk = attack_summary(rec.get("attacks", []))
    if edition == "4e":
        return {
            "name": rec["name"],
            "level": rec.get("level"),
            "role": rec.get("role"),
            "type_line": rec.get("type_line"),
            "hp": rec.get("hp"),
            "ac": rec.get("ac"),
            "fort": rec.get("fort"),
            "ref": rec.get("ref"),
            "will": rec.get("will"),
            "xp": rec.get("xp"),
            **{f"ability_{k}": v for k, v in ab.items()},
            "attack_bonus": atk.get("attack_bonus"),
            "damage_avg": atk.get("damage_avg"),
            "damage_dice": atk.get("damage_dice"),
            "flags": {
                "manual_fix": bool(rec.get("manual_fix")),
                "ocr_suspect": bool(rec.get("ocr_suspect")),
            },
        }
    else:
        saves = rec.get("saves") or {}
        return {
            "name": rec["name"],
            "cr": rec.get("cr"),
            "cr_num": parse_cr(rec.get("cr")),
            "type_line": rec.get("type_line"),
            "hp": rec.get("hp"),
            "ac": rec.get("ac"),
            "xp": rec.get("xp"),
            **{f"ability_{k}": v for k, v in ab.items()},
            **{f"save_{k}": v for k, v in saves.items()},
            "attack_bonus": atk.get("attack_bonus"),
            "damage_avg": atk.get("damage_avg"),
            "damage_dice": atk.get("damage_dice"),
            "flags": {
                "manual_fix": bool(rec.get("manual_fix")),
                "ocr_suspect": bool(rec.get("ocr_suspect")),
            },
        }


def pair_is_plausible(name4: str, name5: str, score: float) -> bool:
    """Filters obvious false positives like Black Dragon -> Blue Dragon."""
    # Very high fuzzy similarity overrides token rules (handles OCR typos).
    if score >= 0.90:
        return True

    core4 = core_tokens(name4)
    core5 = core_tokens(name5)
    if not core4 or not core5:
        return score >= 0.80

    # Require semantic overlap; e.g. "Giant Rat" and "Giant Goat" share no
    # core token, even though their full names are ~0.84 similar.
    if not (core4 & core5):
        return False

    # If both names carry an age descriptor, the ages must align
    # (young->young, adult->adult, elder->ancient, ...).
    age4 = age_canon(name4)
    age5 = age_canon(name5)
    if age4 and age5 and age4 != age5:
        return False

    return score >= 0.55


def match_records(records_4e, records_5e, srd_ref):
    names5 = [r["name"] for r in records_5e]
    by_name5 = {r["name"]: r for r in records_5e}
    by_name4 = {r["name"]: r for r in records_4e}

    pairs = []
    seen = set()

    def add(name4, name5, sc, reason: str):
        key = (name4, name5)
        if key in seen:
            return
        seen.add(key)
        r4 = by_name4[name4]
        r5 = by_name5[name5]
        s4 = build_side(r4, "4e")
        s5 = build_side(r5, "5e")

        unresolved_4e = s4["flags"]["ocr_suspect"] and not s4["flags"]["manual_fix"]
        unresolved_5e = s5["flags"]["ocr_suspect"] and not s5["flags"]["manual_fix"]
        srd_ok = srd_verified(s5["name"], s5["cr"], s5["hp"], srd_ref)
        var_prom = is_variant_promotion(s4["name"], s5["name"], sc)

        # SRD verification is used to trust a 5e block despite an OCR-suspect
        # flag from the parser; it does not bypass the normal score threshold.
        if (unresolved_4e or unresolved_5e) and not srd_ok:
            review = True
        else:
            review = sc < 0.85 and not var_prom

        pairs.append({
            "score": round(sc, 3),
            "reason": reason,
            "review": review,
            "4e": s4,
            "5e": s5,
        })

    norm4 = {r["name"]: normalize(r["name"]) for r in records_4e}
    norm5 = {r["name"]: normalize(r["name"]) for r in records_5e}
    ednorm4 = {r["name"]: edition_normalize(r["name"]) for r in records_4e}
    ednorm5 = {r["name"]: edition_normalize(r["name"]) for r in records_5e}

    # 1. Exact normalized matches.
    n5_to_name = {}
    for r5 in records_5e:
        n5_to_name.setdefault(normalize(r5["name"]), []).append(r5["name"])
    for r4 in records_4e:
        for n5 in n5_to_name.get(normalize(r4["name"]), []):
            add(r4["name"], n5, 1.0, "exact_normalized")

    # 2. Exact edition-normalized matches (Elder -> Ancient dragons, etc.).
    edn5_to_name = {}
    for r5 in records_5e:
        edn5_to_name.setdefault(edition_normalize(r5["name"]), []).append(r5["name"])
    for r4 in records_4e:
        for n5 in edn5_to_name.get(edition_normalize(r4["name"]), []):
            if normalize(r4["name"]) == normalize(n5):
                continue  # already exact
            sc = difflib.SequenceMatcher(None, r4["name"], n5).ratio()
            add(r4["name"], n5, max(sc, 0.99), "exact_edition_normalized")

    # 3. Fuzzy forward scan, gated by core-token and age rules.
    for r4 in records_4e:
        n4 = r4["name"]
        for n5 in names5:
            if n5 in by_name5 and normalize(n5) == normalize(n4):
                continue
            sc = difflib.SequenceMatcher(None, edition_normalize(n4),
                                         edition_normalize(n5)).ratio()
            if pair_is_plausible(n4, n5, sc):
                add(n4, n5, sc, "fuzzy")

    # Sort: high-confidence first, then by score, then names.
    pairs.sort(key=lambda p: (p["review"], -p["score"], p["4e"]["name"], p["5e"]["name"]))
    return pairs


def flatten(prefix: str, side: dict, flat: dict):
    for k, v in side.items():
        if k == "flags":
            flat[f"{prefix}_manual_fix"] = v["manual_fix"]
            flat[f"{prefix}_ocr_suspect"] = v["ocr_suspect"]
        elif isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{prefix}_{kk}"] = vv
        else:
            flat[f"{prefix}_{k}"] = v


def write_outputs(pairs):
    PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in pairs:
        row = {"score": p["score"], "reason": p["reason"], "review": p["review"]}
        flatten("4e", p["4e"], row)
        flatten("5e", p["5e"], row)
        rows.append(row)

    # Logical column order.
    ordered = ["score", "reason", "review"]
    ordered += [f"4e_{k}" for k in [
        "name", "level", "role", "hp", "ac", "fort", "ref", "will", "xp",
        "attack_bonus", "damage_avg", "damage_dice",
        "ability_str", "ability_dex", "ability_con",
        "ability_int", "ability_wis", "ability_cha",
        "type_line", "manual_fix", "ocr_suspect",
    ]]
    ordered += [f"5e_{k}" for k in [
        "name", "cr", "cr_num", "hp", "ac", "xp",
        "attack_bonus", "damage_avg", "damage_dice",
        "ability_str", "ability_dex", "ability_con",
        "ability_int", "ability_wis", "ability_cha",
        "save_str", "save_dex", "save_con", "save_int", "save_wis", "save_cha",
        "type_line", "manual_fix", "ocr_suspect",
    ]]

    # Add any stray keys (defensive).
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    ordered += sorted(k for k in all_keys if k not in ordered)

    with open(PAIRS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    training = [p for p in pairs if not p["review"]]
    with open(TRAINING_PATH, "w") as f:
        for p in training:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    return rows, training


def main():
    with open(MM4_PATH) as f:
        records_4e = [json.loads(line) for line in f]
    with open(MM5_PATH) as f:
        records_5e = [json.loads(line) for line in f]

    srd_ref = load_srd_reference()
    pairs = match_records(records_4e, records_5e, srd_ref)
    rows, training = write_outputs(pairs)

    total = len(pairs)
    high = sum(1 for p in pairs if not p["review"])
    review = total - high
    matched_4e = len({p["4e"]["name"] for p in pairs})
    matched_5e = len({p["5e"]["name"] for p in pairs})

    print(f"4e records: {len(records_4e)}")
    print(f"5e records: {len(records_5e)}")
    print(f"Candidate pairs: {total}")
    print(f"  High confidence (training set): {high}")
    print(f"  Flagged for review: {review}")
    print(f"Unique 4e monsters matched: {matched_4e}")
    print(f"Unique 5e monsters matched: {matched_5e}")
    print(f"Wrote {PAIRS_CSV}")
    print(f"Wrote {TRAINING_PATH}")


if __name__ == "__main__":
    main()
