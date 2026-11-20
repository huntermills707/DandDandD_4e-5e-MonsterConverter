#!/usr/bin/env python3
"""Build cross-edition matched pairs from parsed 4e and 5e stat blocks.

Outputs (local only, derived from the books):
    data/processed/matched_pairs.csv    - accepted pairs plus rejected
                                          candidates, with tier, metadata
                                          score/rank, and a review flag.
    data/processed/training_pairs.jsonl - accepted pairs without unresolved
                                          OCR problems, for regression.
    models/match_scorer.json            - the metadata scorer's coefficients
                                          (tracked; contains no book data).

Matching uses names for evidence and non-numeric metadata as a gate; numeric
stats are never used, so pair selection cannot favor stat relationships the
regressions will later fit. See pipeline/matcher.py.
"""

import csv
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
SCORER_PATH = ROOT / "models" / "match_scorer.json"
RUN_PATH = PROC_DIR / "match_run.json"
CONFIG_PATH = ROOT / "config" / "matching.toml"

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


def load_config(path=CONFIG_PATH, **overrides):
    """config/matching.toml merged with non-None CLI overrides."""
    import tomllib

    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("metadata", {})
    for key, value in overrides.items():
        if value is None:
            continue
        if key in ("top_k", "name_min_sim", "embedding_model"):
            cfg["metadata"][key] = value
        else:
            cfg[key] = value
    if cfg["method"] not in ("metadata", "legacy"):
        raise ValueError(f"unknown method {cfg['method']!r}")
    return cfg


def match_records(records_4e, records_5e, srd_ref, cfg):
    """Run the configured matcher, flag data problems, apply cardinality.
    Returns (pairs, scorer summary or None)."""
    from pipeline.cardinality import select

    scorer = None
    if cfg["method"] == "legacy":
        from pipeline.matcher_legacy import match as legacy_match

        accepted, rejected = legacy_match([r["name"] for r in records_4e],
                                          [r["name"] for r in records_5e])
    else:
        from pipeline.matcher import match, scorer_summary
        from pipeline.metadata import profile_4e, profile_5e

        meta = cfg["metadata"]
        accepted, rejected, model = match(
            [profile_4e(r) for r in records_4e], [profile_5e(r) for r in records_5e],
            top_k=meta["top_k"], name_min_sim=meta["name_min_sim"],
            model_name=meta["embedding_model"])
        scorer = scorer_summary(model)

    def data_problem(m):
        s4 = build_side(records_4e[m["i4"]], "4e")
        s5 = build_side(records_5e[m["i5"]], "5e")
        unresolved_4e = s4["flags"]["ocr_suspect"] and not s4["flags"]["manual_fix"]
        unresolved_5e = s5["flags"]["ocr_suspect"] and not s5["flags"]["manual_fix"]
        # SRD verification lets a 5e block be trusted despite an OCR-suspect
        # flag from the parser.
        return (unresolved_4e or unresolved_5e) and not srd_verified(
            s5["name"], s5["cr"], s5["hp"], srd_ref)

    trainable = [m for m in accepted if not data_problem(m)]
    flagged = [dict(m, note="unresolved OCR") for m in accepted if data_problem(m)]
    kept, dropped = select(trainable, cfg["cardinality"], cfg["elbow_ratio"])
    dropped = [dict(m, note=f"dropped by cardinality={cfg['cardinality']}") for m in dropped]

    pairs = []
    for m, is_accepted, review in ([(m, True, False) for m in kept]
                                   + [(m, True, True) for m in flagged + dropped]
                                   + [(m, False, True) for m in rejected]):
        pairs.append({
            "reason": m["tier"],
            "match_type": m["match_type"],
            "score": m["score"],
            "meta_prob": m["meta_prob"],
            "meta_rank": m["meta_rank"],
            "name_sim": m["name_sim"],
            "accepted": is_accepted,
            "review": review,
            "note": m["note"],
            "4e": build_side(records_4e[m["i4"]], "4e"),
            "5e": build_side(records_5e[m["i5"]], "5e"),
        })
    # Training order feeds the seeded model splits; keep it deterministic.
    pairs.sort(key=lambda p: (p["review"], -p["score"], p["4e"]["name"], p["5e"]["name"]))
    return pairs, scorer


def assign_groups(training):
    """Give each pair a CV group: connected components of the 4e/5e monster
    graph, so pairs sharing a monster (variants of one 5e base, a 4e monster
    with two 5e matches) never straddle train and test. Ids are numbered by
    first appearance and carry no names."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in training:
        a, b = find(("4e", p["4e"]["name"])), find(("5e", p["5e"]["name"]))
        parent[a] = b
    ids = {}
    for p in training:
        p["group"] = ids.setdefault(find(("5e", p["5e"]["name"])), len(ids))


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
        row = {k: p[k] for k in ("reason", "match_type", "score", "meta_prob", "meta_rank",
                                 "name_sim", "accepted", "review", "note")}
        flatten("4e", p["4e"], row)
        flatten("5e", p["5e"], row)
        rows.append(row)

    # Logical column order.
    ordered = ["reason", "match_type", "score", "meta_prob", "meta_rank", "name_sim",
               "accepted", "review", "note"]
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
    assign_groups(training)
    with open(TRAINING_PATH, "w") as f:
        for p in training:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    return rows, training


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Build cross-edition matched pairs.")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--method", choices=["metadata", "legacy"])
    ap.add_argument("--cardinality", choices=["many", "one", "elbow"])
    ap.add_argument("--elbow-ratio", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--name-min-sim", type=float)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, method=args.method, cardinality=args.cardinality,
                      elbow_ratio=args.elbow_ratio, top_k=args.top_k,
                      name_min_sim=args.name_min_sim)
    print(f"method={cfg['method']} cardinality={cfg['cardinality']}"
          + (f" elbow_ratio={cfg['elbow_ratio']}" if cfg["cardinality"] == "elbow" else ""))

    with open(MM4_PATH) as f:
        records_4e = [json.loads(line) for line in f]
    with open(MM5_PATH) as f:
        records_5e = [json.loads(line) for line in f]

    srd_ref = load_srd_reference()
    pairs, scorer = match_records(records_4e, records_5e, srd_ref, cfg)
    rows, training = write_outputs(pairs)
    if scorer is not None:
        with open(SCORER_PATH, "w", encoding="utf-8") as f:
            json.dump(scorer, f, indent=2)
    run = {k: cfg[k] for k in ("method", "cardinality", "elbow_ratio")}
    if cfg["method"] == "metadata":
        run["metadata"] = cfg["metadata"]
    run["n_training_pairs"] = len(training)
    with open(RUN_PATH, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    from collections import Counter

    total = len(pairs)
    high = len(training)
    review = total - high
    matched_4e = len({p["4e"]["name"] for p in training})
    matched_5e = len({p["5e"]["name"] for p in training})

    print(f"4e records: {len(records_4e)}")
    print(f"5e records: {len(records_5e)}")
    print(f"Candidate pairs: {total}")
    print(f"  High confidence (training set): {high}")
    print(f"  Flagged for review: {review}")
    print(f"  Training pairs by tier: {dict(Counter(p['reason'] for p in training))}")
    print(f"Unique 4e monsters in training: {matched_4e}")
    print(f"Unique 5e monsters in training: {matched_5e}")
    print(f"Wrote {PAIRS_CSV}")
    print(f"Wrote {TRAINING_PATH}")


if __name__ == "__main__":
    main()
