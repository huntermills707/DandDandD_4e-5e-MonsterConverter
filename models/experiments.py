#!/usr/bin/env python3
"""Compare model designs under the same repeated grouped CV as fit_models.py.

Each design maps a model spec to (features, LinearDesign). Metrics are on
held-out same-creature pairs: MAE relative to the mean baseline (lower is
better), R², 80% interval coverage, and mean interval width relative to the
true value.

Usage:
    .venv/bin/python -m models.experiments [--data real|surrogate] [--fields]
"""

import argparse

import numpy as np

from models.fit_models import MODEL_SPECS, LinearDesign, fit_spec
from pipeline.datasets import SOURCES, load_pairs, resolve_source

# Feature lists of the original specs (designs D0-D7), kept for comparison.
FULL_SPECS = [
    # 4e -> 5e
    {"direction": "4e_to_5e", "target": "cr_num", "features": ["level", "hp", "ac", "attack_bonus", "damage_avg"]},
    {"direction": "4e_to_5e", "target": "hp", "features": ["hp", "level", "ac", "ability_con"]},
    {"direction": "4e_to_5e", "target": "ac", "features": ["ac", "level", "ability_dex"]},
    {"direction": "4e_to_5e", "target": "attack_bonus", "features": ["attack_bonus", "level", "ability_str", "ability_dex"]},
    {"direction": "4e_to_5e", "target": "damage_avg", "features": ["damage_avg", "level", "ability_str"]},
    {"direction": "4e_to_5e", "target": "ability_str", "features": ["ability_str", "level"]},
    {"direction": "4e_to_5e", "target": "ability_dex", "features": ["ability_dex", "level"]},
    {"direction": "4e_to_5e", "target": "ability_con", "features": ["ability_con", "level"]},
    {"direction": "4e_to_5e", "target": "ability_int", "features": ["ability_int", "level"]},
    {"direction": "4e_to_5e", "target": "ability_wis", "features": ["ability_wis", "level"]},
    {"direction": "4e_to_5e", "target": "ability_cha", "features": ["ability_cha", "level"]},
    {"direction": "4e_to_5e", "target": "save_dex", "features": ["ref", "ability_dex", "level"]},
    {"direction": "4e_to_5e", "target": "save_con", "features": ["fort", "ability_con", "level"]},
    {"direction": "4e_to_5e", "target": "save_wis", "features": ["will", "ability_wis", "level"]},
    {"direction": "4e_to_5e", "target": "save_cha", "features": ["will", "ability_cha", "level"]},

    # 5e -> 4e
    {"direction": "5e_to_4e", "target": "level", "features": ["cr_num", "hp", "ac", "attack_bonus", "damage_avg"]},
    {"direction": "5e_to_4e", "target": "hp", "features": ["hp", "cr_num", "ac", "ability_con"]},
    {"direction": "5e_to_4e", "target": "ac", "features": ["ac", "cr_num", "ability_dex"]},
    {"direction": "5e_to_4e", "target": "fort", "features": ["ability_con", "ability_str", "cr_num", "hp"]},
    {"direction": "5e_to_4e", "target": "ref", "features": ["ability_dex", "cr_num", "hp"]},
    {"direction": "5e_to_4e", "target": "will", "features": ["ability_wis", "ability_cha", "cr_num", "hp"]},
    {"direction": "5e_to_4e", "target": "attack_bonus", "features": ["attack_bonus", "cr_num", "ability_str", "ability_dex"]},
    {"direction": "5e_to_4e", "target": "damage_avg", "features": ["damage_avg", "cr_num", "ability_str"]},
    {"direction": "5e_to_4e", "target": "ability_str", "features": ["ability_str", "cr_num"]},
    {"direction": "5e_to_4e", "target": "ability_dex", "features": ["ability_dex", "cr_num"]},
    {"direction": "5e_to_4e", "target": "ability_con", "features": ["ability_con", "cr_num"]},
    {"direction": "5e_to_4e", "target": "ability_int", "features": ["ability_int", "cr_num"]},
    {"direction": "5e_to_4e", "target": "ability_wis", "features": ["ability_wis", "cr_num"]},
    {"direction": "5e_to_4e", "target": "ability_cha", "features": ["ability_cha", "cr_num"]},
]
FULL_FEATURES = {(s["direction"], s["target"]): s["features"] for s in FULL_SPECS}

ROLE = ["role_minion", "role_elite", "role_solo"]
MULTIPLICATIVE = {"hp", "damage_avg"}  # scale multiplicatively in both editions
LEVEL_LIKE = {"cr_num", "level"}


def d0(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    return spec["features"], LinearDesign("D0 nonneg/relative")


def d1(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    return spec["features"], LinearDesign("D1 ols/absolute", fit="ols", interval="absolute")


def d2(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    return spec["features"] + ROLE, LinearDesign("D2 +role", fit="ols", interval="absolute")


def d3(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    feats = spec["features"] + ROLE
    if spec["target"] in MULTIPLICATIVE:
        return feats, LinearDesign("D3 +role, log hp/dmg", fit="ols", interval="log",
                                   log_target=True, log_features=MULTIPLICATIVE)
    return feats, LinearDesign("D3 +role, log hp/dmg", fit="ols", interval="absolute",
                               log_features=MULTIPLICATIVE)


def d4(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    feats = spec["features"] + ROLE
    if spec["target"] in MULTIPLICATIVE | LEVEL_LIKE:
        return feats, LinearDesign("D4 D3 + log CR/level", fit="ols", interval="log",
                                   log_target=True, log_features=MULTIPLICATIVE | LEVEL_LIKE)
    return feats, LinearDesign("D4 D3 + log CR/level", fit="ols", interval="absolute",
                               log_features=MULTIPLICATIVE | LEVEL_LIKE)


def d5(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    return spec["features"] + ROLE, LinearDesign("D5 +role, ridge", fit="ridge", interval="absolute")


def d6(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    feats = spec["features"] + ROLE
    if spec["target"] in MULTIPLICATIVE:
        return feats, LinearDesign("D6 +role, ridge, log hp/dmg", fit="ridge", interval="log",
                                   log_target=True, log_features=MULTIPLICATIVE)
    return feats, LinearDesign("D6 +role, ridge, log hp/dmg", fit="ridge", interval="absolute",
                               log_features=MULTIPLICATIVE)


def d7(spec):
    spec = dict(spec, features=FULL_FEATURES[(spec["direction"], spec["target"])])
    return spec["features"] + ROLE, LinearDesign("D7 +role, scaled interval", fit="ols",
                                                 interval="scaled")


def minimal_features(spec):
    return spec["features"]  # MODEL_SPECS: counterpart stat(s) + level/CR


def d8(spec):
    return minimal_features(spec) + ROLE, LinearDesign("D8 minimal +role, scaled", fit="ols",
                                                       interval="scaled")


def d9(spec):
    return minimal_features(spec) + ROLE, LinearDesign("D9 D8, 2-sig-fig coefs", fit="ols",
                                                       interval="scaled", round_sig=2)


def d10(spec):
    return minimal_features(spec) + ROLE, LinearDesign("D10 D8, 1-sig-fig coefs", fit="ols",
                                                        interval="scaled", round_sig=1)


DESIGNS = [d0, d2, d7, d8, d9, d10]


def nested_selection(pairs, n_repeats=3):
    """Honest cost of feature pruning: selection re-run inside each outer
    training fold, vs. the unpruned explainable design. Returns rows of
    (field, unpruned MAE/baseline, pruned MAE/baseline)."""
    from models.fit_models import (COUNTERPART, DEFAULT_DESIGN, ROLE_FEATURES, extract,
                                   grouped_folds, select_features, source_target)

    rows = []
    for spec in MODEL_SPECS:
        src, tgt = source_target(spec)
        full = spec["features"] + ROLE_FEATURES
        X, y, g, same = extract(pairs, src, tgt, full, spec["target"])
        usable = [p for p in pairs if p[tgt].get(spec["target"]) is not None and all(
            f.startswith("role_") or p[src].get(f) is not None for f in full)]
        ratios = {"full": [], "pruned": []}
        for rep in range(n_repeats):
            pred = {"full": np.empty_like(y), "pruned": np.empty_like(y)}
            base = np.empty_like(y)
            for test in grouped_folds(g, 5, 2000 + rep):
                tr = np.setdiff1d(np.arange(len(y)), test)
                m = DEFAULT_DESIGN()
                m.features = full
                pred["full"][test] = m.fit(X[tr], y[tr]).predict(X[test])
                kept, _ = select_features([usable[k] for k in tr], dict(spec, features=full),
                                          DEFAULT_DESIGN, COUNTERPART[spec["direction"]][spec["target"]])
                idx = [full.index(f) for f in kept]
                ms = DEFAULT_DESIGN()
                ms.features = kept
                pred["pruned"][test] = ms.fit(X[tr][:, idx], y[tr]).predict(X[test][:, idx])
                base[test] = y[tr].mean()
            b = np.abs(y - base)[same].mean()
            for k in ratios:
                ratios[k].append(np.abs(y - pred[k])[same].mean() / b)
        rows.append((f"{spec['direction']}/{spec['target']}",
                     float(np.mean(ratios["full"])), float(np.mean(ratios["pruned"]))))
    return rows


def run(pairs, design):
    out = {}
    for spec in MODEL_SPECS:
        feats, model = design(spec)
        r = fit_spec(pairs, dict(spec, features=feats), make_model=model)
        if r is not None:
            out[f"{spec['direction']}/{spec['target']}"] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", choices=SOURCES, default="auto")
    ap.add_argument("--fields", action="store_true", help="per-field MAE ratio table")
    ap.add_argument("--nested-selection", action="store_true",
                    help="honest cost of feature pruning (selection inside each fold)")
    args = ap.parse_args(argv)
    pairs = load_pairs(resolve_source(args.data))

    if args.nested_selection:
        rows = nested_selection(pairs)
        for key, full, pruned in rows:
            print(f"{key:26s} unpruned {full:.3f}   pruned {pruned:.3f}")
        print(f"median MAE/baseline: unpruned {np.median([r[1] for r in rows]):.3f}, "
              f"pruned (selection nested) {np.median([r[2] for r in rows]):.3f}")
        return

    results = {d.__name__: run(pairs, d) for d in DESIGNS}
    print(f"{'design':28s} {'MAE/base':>8s} {'beat':>6s} {'medR2':>6s} {'cov':>5s} "
          f"{'worst3rd':>8s} {'width':>6s}")
    for name, res in results.items():
        ratio = [m["mae"] / m["baseline_mae"] for m in res.values()]
        label = next(iter(res.values()))["estimator"]
        worst = [min(m["coverage_by_third"]) for m in res.values()]
        print(f"{label:28s} {np.median(ratio):8.3f} {sum(r < 1 for r in ratio):3d}/{len(ratio):<2d} "
              f"{np.median([m['r2'] for m in res.values()]):6.2f} "
              f"{np.mean([m['test_coverage'] for m in res.values()]):5.2f} "
              f"{np.median(worst):8.2f} "
              f"{np.median([m['interval_rel_width'] for m in res.values()]):6.2f}")
    if args.fields:
        print(f"\n{'field: worst-third coverage':26s} " + " ".join(f"{n:>6s}" for n in results))
        for key in results["d0"]:
            print(f"{key:26s} " + " ".join(
                f"{min(results[n][key]['coverage_by_third']):6.2f}" for n in results))
        names = list(results)
        print(f"\n{'field (MAE / baseline)':26s} " + " ".join(f"{n:>6s}" for n in names))
        for key in results["d0"]:
            row = [results[n][key]["mae"] / results[n][key]["baseline_mae"] for n in names]
            best = int(np.argmin(row))
            print(f"{key:26s} " + " ".join(
                f"{v:5.2f}{'*' if i == best else ' '}" for i, v in enumerate(row)))


if __name__ == "__main__":
    main()
