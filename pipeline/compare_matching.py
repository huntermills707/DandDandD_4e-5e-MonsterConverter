#!/usr/bin/env python3
"""Compare matching configurations by the conversion models they produce.

Every configuration (method x cardinality) yields a different training set,
so each fit_models.py report is scored on a different test split and the
numbers aren't comparable. Here all configurations share one test set:
same-creature pairs that every configuration accepts, split into 5 folds
grouped by 5e monster. For each fold, a configuration trains on its own
pairs minus any pair whose 4e or 5e monster is in the fold (a "War Troll ->
Troll" pair would otherwise leak the 5e Troll's stats into the test).

Usage:
    .venv/bin/python -m pipeline.compare_matching
"""

import itertools
import json

import numpy as np
from sklearn.linear_model import LinearRegression

from models.fit_models import MODEL_SPECS, extract, source_target
from pipeline.build_pairs import MM4_PATH, MM5_PATH, load_config, load_srd_reference, match_records

N_FOLDS = 5
SEED = 0


def training_pairs(cfg, records_4e, records_5e, srd_ref):
    pairs, _ = match_records(records_4e, records_5e, srd_ref, cfg)
    return [p for p in pairs if not p["review"]]


def key(p):
    return p["4e"]["name"], p["5e"]["name"]


def fit_predict(train, test, spec):
    src, tgt = source_target(spec)
    Xtr, ytr = extract(train, src, tgt, spec["features"], spec["target"])
    Xte, yte = extract(test, src, tgt, spec["features"], spec["target"])
    if len(ytr) < 15 or len(yte) == 0:
        return None
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(np.column_stack([np.ones(len(ytr)), Xtr]), ytr)
    pred = model.predict(np.column_stack([np.ones(len(yte)), Xte]))
    return yte, pred, np.full_like(yte, ytr.mean())


def evaluate(config_pairs, shared_test):
    """Out-of-fold predictions per field for each configuration."""
    test_5e = sorted({p["5e"]["name"] for p in shared_test})
    rng = np.random.default_rng(SEED)
    fold_of = {n: f for n, f in zip(rng.permutation(test_5e), itertools.cycle(range(N_FOLDS)))}

    results = {name: {} for name in config_pairs}
    for fold in range(N_FOLDS):
        test = [p for p in shared_test if fold_of[p["5e"]["name"]] == fold]
        held_4e = {p["4e"]["name"] for p in test}
        held_5e = {p["5e"]["name"] for p in test}
        for name, pairs in config_pairs.items():
            train = [p for p in pairs
                     if p["4e"]["name"] not in held_4e and p["5e"]["name"] not in held_5e]
            for spec in MODEL_SPECS:
                out = fit_predict(train, test, spec)
                if out is None:
                    continue
                field = f"{spec['direction']}/{spec['target']}"
                acc = results[name].setdefault(field, ([], [], []))
                for lst, arr in zip(acc, out):
                    lst.extend(arr)
    summary = {}
    for name, fields in results.items():
        r2s, ratios = [], []
        for y, pred, base in (map(np.array, v) for v in fields.values()):
            ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
            r2s.append(1 - ss_res / ss_tot if ss_tot else np.nan)
            ratios.append(np.abs(y - pred).mean() / np.abs(y - base).mean())
        summary[name] = {
            "median_r2": float(np.nanmedian(r2s)),
            "median_mae_vs_baseline": float(np.median(ratios)),
            "beat_baseline": int(sum(r < 1 for r in ratios)),
            "n_fields": len(ratios),
        }
    return summary


def main():
    with open(MM4_PATH) as f:
        records_4e = [json.loads(line) for line in f]
    with open(MM5_PATH) as f:
        records_5e = [json.loads(line) for line in f]
    srd_ref = load_srd_reference()

    config_pairs = {}
    for method, card in itertools.product(["legacy", "metadata"], ["many", "elbow", "one"]):
        cfg = load_config(method=method, cardinality=card)
        config_pairs[f"{method}/{card}"] = training_pairs(cfg, records_4e, records_5e, srd_ref)

    same_sets = [{key(p) for p in pairs if p["match_type"] == "same"}
                 for pairs in config_pairs.values()]
    shared_keys = set.intersection(*same_sets)
    ref = config_pairs["legacy/many"] + config_pairs["metadata/many"]
    shared_test = list({key(p): p for p in ref if key(p) in shared_keys}.values())
    print(f"shared test pairs (same creature in every config): {len(shared_test)}, "
          f"{len({p['5e']['name'] for p in shared_test})} 5e monsters, {N_FOLDS} grouped folds\n")

    summary = evaluate(config_pairs, shared_test)
    print(f"{'config':18s} {'pairs':>5s} {'same':>5s} {'rel':>4s} {'5e':>4s} "
          f"{'max/5e':>6s} {'medR2':>6s} {'MAE/base':>8s} {'beat':>6s}")
    for name, pairs in config_pairs.items():
        per5 = {}
        for p in pairs:
            per5[p["5e"]["name"]] = per5.get(p["5e"]["name"], 0) + 1
        same = sum(p["match_type"] == "same" for p in pairs)
        s = summary[name]
        print(f"{name:18s} {len(pairs):5d} {same:5d} {len(pairs) - same:4d} {len(per5):4d} "
              f"{max(per5.values()):6d} {s['median_r2']:6.2f} {s['median_mae_vs_baseline']:8.2f} "
              f"{s['beat_baseline']:3d}/{s['n_fields']}")


if __name__ == "__main__":
    main()
