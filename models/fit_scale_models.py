#!/usr/bin/env python3
"""Fit within-edition stat-scaling curves: stat ~ level (4e) / CR (5e).

These curves power the difficulty scaler app (app/scale_app.py): a monster's
stats are shifted along the edition's own curve when moving to a new level/CR.

Two model kinds per stat:
    additive       stat = intercept + slope * axis        (new = old + slope*Δaxis)
    multiplicative log(stat) = intercept + slope * axis   (new = old * exp(slope*Δaxis))

HP and damage grow multiplicatively in both editions; everything else is
treated additively.  Results are saved to models/scale_models.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.datasets import SOURCES, load_sides, resolve_source  # noqa: E402

MODELS_DIR = ROOT / "models"

MIN_N = 30

MULTIPLICATIVE = {"hp", "damage_avg", "damage_per_round"}

EDITIONS = {
    "4e": {
        "axis": "level",
        "stats": [
            "hp", "ac", "attack_bonus", "damage_avg",
            "ability_str", "ability_dex", "ability_con",
            "ability_int", "ability_wis", "ability_cha",
            "fort", "ref", "will",
        ],
    },
    "5e": {
        "axis": "cr_num",
        "stats": [
            "hp", "ac", "attack_bonus", "damage_avg", "damage_per_round",
            "ability_str", "ability_dex", "ability_con",
            "ability_int", "ability_wis", "ability_cha",
            "save_str", "save_dex", "save_con",
            "save_int", "save_wis", "save_cha",
        ],
    },
}


def fit_one(xs, ys, kind):
    """OLS of ys on xs; for 'multiplicative', OLS of log(ys) on xs."""
    X = np.array(xs, dtype=float).reshape(-1, 1)
    y = np.array(ys, dtype=float)
    if kind == "multiplicative":
        mask = y > 0
        X, y = X[mask], y[mask]
        if len(y) < MIN_N:
            return None
        y = np.log(y)
    if len(y) < MIN_N:
        return None
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    return {
        "kind": kind,
        "slope": round(float(model.coef_[0]), 6),
        "intercept": round(float(model.intercept_), 4),
        "per_step_ratio": round(float(math.exp(model.coef_[0])), 6) if kind == "multiplicative" else None,
        "r2": round(float(r2_score(y, pred)), 3),
        "n": int(len(y)),
    }


def fit_edition(edition, spec, source):
    sides = load_sides(edition, source)
    axis = spec["axis"]
    rows = [s for s in sides if s.get(axis) is not None]

    # Pool every available save value into one generic save curve; used for
    # save stats too sparse to fit individually (5e STR/INT saves).
    pooled_x, pooled_y = [], []
    for s in rows:
        for save in ("save_str", "save_dex", "save_con", "save_int", "save_wis", "save_cha"):
            if s.get(save) is not None:
                pooled_x.append(s[axis])
                pooled_y.append(s[save])
    pooled_save = fit_one(pooled_x, pooled_y, "additive") if pooled_x else None

    stats = {}
    for stat in spec["stats"]:
        kind = "multiplicative" if stat in MULTIPLICATIVE else "additive"
        xs = [s[axis] for s in rows if s.get(stat) is not None]
        ys = [s[stat] for s in rows if s.get(stat) is not None]
        result = fit_one(xs, ys, kind)
        if result is None and stat.startswith("save_") and pooled_save is not None:
            result = dict(pooled_save)
            result["pooled"] = True
        if result is None:
            print(f"  {edition}/{stat}: skipped (n < {MIN_N})")
            continue
        stats[stat] = result
        print(f"  {edition}/{stat}: {kind}, slope={result['slope']}, "
              f"R²={result['r2']}, n={result['n']}"
              + (" [pooled]" if result.get("pooled") else ""))

    return {
        "axis": axis,
        "n_monsters": len(rows),
        "stats": stats,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=SOURCES, default="auto",
                        help="real local data, tracked surrogate, or auto (real if present)")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    source = resolve_source(args.data)
    print(f"Data source: {source}")
    out = {}
    for edition, spec in EDITIONS.items():
        print(f"{edition}:")
        out[edition] = fit_edition(edition, spec, source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "scale_models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
