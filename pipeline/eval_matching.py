#!/usr/bin/env python3
"""Evaluate the matcher.

Part 1, name-blind retrieval of the metadata model:

Ground truth: 4e/5e records whose (edition-normalized) names are identical
are the same creature. For each such 4e monster we hide all names, rank all
5e monsters by a metadata-only score, and report where the true counterpart
lands. Scorers are trained with grouped cross-validation (folds split by 4e
monster) so a monster is never ranked by a model that saw its label.

Part 2, end-to-end decisions (needs manual/local/match_labels.csv): the
matcher's accepted non-exact pairs are scored against hand labels of
candidate pairs (label 1 = same creature or the direct base/analog a GM
would convert to, 0 = different creature, blank = unsure). Recall is over
4e monsters that have a labeled-correct candidate.

Usage:
    .venv/bin/python -m pipeline.eval_matching [--model MODEL_NAME]
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pipeline.datasets import REAL_RAW
from pipeline.match_features import (
    DEFAULT_MODEL, META_FEATURES, meta_matrix, norm_name,
)
from pipeline.metadata import profile_4e, profile_5e


def load_profiles():
    with open(REAL_RAW["4e"], encoding="utf-8") as f:
        r4 = [json.loads(line) for line in f]
    with open(REAL_RAW["5e"], encoding="utf-8") as f:
        r5 = [json.loads(line) for line in f]
    return [profile_4e(r) for r in r4], [profile_5e(r) for r in r5]


def gold_pairs(p4s, p5s):
    idx5 = {}
    for j, p in enumerate(p5s):
        idx5.setdefault(norm_name(p["name"]), []).append(j)
    return {i: idx5[norm_name(p["name"])] for i, p in enumerate(p4s)
            if norm_name(p["name"]) in idx5}


def rank_metrics(scores, gold):
    """scores: {i4: array over 5e}; gold: {i4: [j5, ...]} -> metrics."""
    ranks = []
    for i, js in gold.items():
        order = np.argsort(-scores[i])
        pos = np.empty_like(order)
        pos[order] = np.arange(len(order))
        ranks.append(1 + min(pos[j] for j in js))
    ranks = np.array(ranks)
    return {
        "n": len(ranks),
        "recall@1": round(float(np.mean(ranks <= 1)), 3),
        "recall@5": round(float(np.mean(ranks <= 5)), 3),
        "recall@10": round(float(np.mean(ranks <= 10)), 3),
        "MRR": round(float(np.mean(1 / ranks)), 3),
        "median_rank": float(np.median(ranks)),
    }


def cv_scores(X, gold, cols, n_splits=5, seed=0):
    """Out-of-fold logistic-regression scores for every gold 4e row."""
    rows = np.array(sorted(gold))
    rng = np.random.default_rng(seed)
    groups = rng.permutation(len(rows))
    out = {}
    for tr, te in GroupKFold(n_splits).split(rows, groups=groups):
        Xs, ys = [], []
        for i in rows[tr]:
            y = np.zeros(X.shape[1])
            y[gold[i]] = 1
            Xs.append(X[i][:, cols])
            ys.append(y)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(np.vstack(Xs), np.concatenate(ys))
        for i in rows[te]:
            out[i] = clf.decision_function(X[i][:, cols])
    return out


LABELS_PATH = REAL_RAW["4e"].parents[2] / "manual" / "local" / "match_labels.csv"


def evaluate_decisions():
    import csv

    from pipeline.matcher import match

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = {(r["name_4e"], r["name_5e"]): (int(r["label"]) if r["label"] else None)
                  for r in csv.DictReader(f)}
    with open(REAL_RAW["4e"], encoding="utf-8") as f:
        names4 = [json.loads(line)["name"] for line in f]
    with open(REAL_RAW["5e"], encoding="utf-8") as f:
        names5 = [json.loads(line)["name"] for line in f]

    p4s, p5s = load_profiles()
    accepted, _, _ = match(p4s, p5s)
    model_pairs = [(names4[m["i4"]], names5[m["i5"]]) for m in accepted if m["tier"] == "model"]
    verdicts = [labels.get(pair, "unlabeled") for pair in model_pairs]
    tp, fp = verdicts.count(1), verdicts.count(0)
    positives = {a for (a, _), v in labels.items() if v == 1}
    print(f"\nModel-proposed pairs: {len(model_pairs)}  (right {tp}, wrong {fp}, "
          f"unsure {verdicts.count(None)}, unlabeled {verdicts.count('unlabeled')})")
    print(f"precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(len(positives), 1):.2f} "
          f"over {len(positives)} 4e monsters with a labeled-correct candidate")
    for pair, v in zip(model_pairs, verdicts):
        if v != 1:
            print(f"   [{v}] {pair[0]} -> {pair[1]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)

    p4s, p5s = load_profiles()
    gold = gold_pairs(p4s, p5s)
    print(f"{len(p4s)} 4e x {len(p5s)} 5e records; {len(gold)} 4e monsters with a same-name 5e counterpart")
    print(f"chance recall@1 = {1 / len(p5s):.3f}, recall@10 = {10 / len(p5s):.3f}\n")

    X = meta_matrix(p4s, p5s, args.model)
    f = {name: k for k, name in enumerate(META_FEATURES)}

    results = {
        "embedding cosine only": {i: X[i][:, f["emb_cos"]] for i in gold},
        "attack-name overlap only": {i: X[i][:, f["attacks_jacc"]] for i in gold},
        "block text TF-IDF only": {i: X[i][:, f["text_tfidf_cos"]] for i in gold},
        "block text embedding only": {i: X[i][:, f["text_emb_cos"]] for i in gold},
        "structured features (LR, no embedding/text)": cv_scores(
            X, gold, [k for n, k in f.items() if n not in ("emb_cos", "text_tfidf_cos", "text_emb_cos")]),
        "all metadata (LR)": cv_scores(X, gold, list(f.values())),
    }
    print(f"{'scorer':42s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'med':>5s}")
    for name, scores in results.items():
        m = rank_metrics(scores, gold)
        print(f"{name:42s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} "
              f"{m['recall@10']:6.3f} {m['MRR']:6.3f} {m['median_rank']:5.0f}")

    if LABELS_PATH.exists():
        evaluate_decisions()
    else:
        print(f"\n(no {LABELS_PATH.name}; skipping decision evaluation)")


if __name__ == "__main__":
    main()
