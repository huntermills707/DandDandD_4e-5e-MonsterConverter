"""Decide 4e -> 5e pairs.

1. Approximately exact names -> match ("same" creature):
       exact  same name after normalization, including OCR word splits
              ("Adult S Ilver Dragon") and 4e Elder == 5e Ancient
       alias  a parenthetical alternate name matches ("Wailing Ghost
              (Banshee)" -> "Banshee")
       ocr    near-identical spelling (>= 0.9 similarity, spaces removed)
2. Otherwise the metadata model proposes: its TOP_K best-scoring 5e
   monsters are tried best-first, and the first whose name similarity
   (IDF-weighted fuzzy ROUGE-1 recall, pipeline/name_similarity.py) reaches
   NAME_MIN_SIM is accepted as a "related" pair (variant or analog: "Horde
   Ghoul" -> "Ghoul", "Fire Titan" -> "Fire Giant"). Dragon ages must agree.

The metadata model is a logistic regression over non-numeric features
(pipeline/match_features.py), trained on exact-name pairs. TOP_K and
NAME_MIN_SIM were chosen on hand-labeled candidates (manual/local/
match_labels.csv): 98% precision, 82% recall; see pipeline/eval_matching.py.
Taking only the single top candidate (TOP_K = 1) keeps precision but finds
far fewer matches (~40% recall), because a sibling ("Goblin Boss") often
scores just above the base creature ("Goblin").
"""

import difflib
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.match_features import DEFAULT_MODEL, META_FEATURES, meta_matrix, norm_name
from pipeline.name_similarity import NameSimilarity

TOP_K = 5
NAME_MIN_SIM = 0.65
NEAR_MISS_SIM = 0.4  # model candidates at least this similar go to review
OCR_MIN_RATIO = 0.9

AGES = {"wyrmling", "young", "adult", "ancient"}
SAME_TIERS = ("exact", "alias", "ocr")


def despace(name):
    return re.sub(r"[^a-z]", "", norm_name(name))


def name_tier(p4, p5):
    """'exact' | 'alias' | 'ocr' | None for approximately exact names."""
    n4, n5 = norm_name(p4["name"]), norm_name(p5["name"])
    if n4 == n5 or despace(p4["name"]) == despace(p5["name"]):
        return "exact"
    names4 = {n4} | {norm_name(a) for a in p4["alt_names"]}
    names5 = {n5} | {norm_name(a) for a in p5["alt_names"]}
    if names4 & names5:
        return "alias"
    if difflib.SequenceMatcher(None, despace(p4["name"]), despace(p5["name"])).ratio() >= OCR_MIN_RATIO:
        return "ocr"
    return None


def ages_conflict(name4, name5):
    a4 = set(norm_name(name4).split()) & AGES
    a5 = set(norm_name(name5).split()) & AGES
    return bool(a4 and a5 and a4 != a5)


def fit_scorer(X, gold):
    """Logistic regression on exact-name pairs; returns the fitted model."""
    rows = sorted(gold)
    Xs = np.vstack([X[i] for i in rows])
    ys = np.concatenate([np.isin(np.arange(X.shape[1]), gold[i]).astype(float) for i in rows])
    return LogisticRegression(max_iter=3000).fit(Xs, ys)


def match(p4s, p5s, top_k=TOP_K, name_min_sim=NAME_MIN_SIM, model_name=DEFAULT_MODEL):
    """Return (accepted, review, scorer); accepted/review are lists of dicts
    {i4, i5, tier, match_type, score, meta_prob, meta_rank, name_sim, note}
    where score is the metadata probability."""
    X = meta_matrix(p4s, p5s, model_name)
    tiers = [[name_tier(a, b) for b in p5s] for a in p4s]
    gold = {i: [j for j, t in enumerate(row) if t == "exact"] for i, row in enumerate(tiers)}
    gold = {i: js for i, js in gold.items() if js}
    scorer = fit_scorer(X, gold)
    P = np.stack([scorer.predict_proba(X[i])[:, 1] for i in range(len(p4s))])
    R = (-P).argsort(axis=1).argsort(axis=1) + 1

    names = NameSimilarity([p["name"] for p in p4s + p5s])

    def sim(i, j):
        if ages_conflict(p4s[i]["name"], p5s[j]["name"]):
            return 0.0
        return names.rouge1_recall(p4s[i]["name"], p5s[j]["name"])

    def rec(i, j, tier, match_type, note=""):
        return {"i4": i, "i5": j, "tier": tier, "match_type": match_type,
                "score": round(float(P[i, j]), 4),
                "meta_prob": round(float(P[i, j]), 4), "meta_rank": int(R[i, j]),
                "name_sim": round(sim(i, j), 3), "note": note}

    accepted, review = [], []
    for i in range(len(p4s)):
        best = next((t for t in SAME_TIERS if t in tiers[i]), None)
        if best:
            accepted.extend(rec(i, j, best, "same")
                            for j, t in enumerate(tiers[i]) if t == best)
            continue

        top = np.argsort(-P[i])[:top_k]
        hit = next((int(j) for j in top if sim(i, j) >= name_min_sim), None)
        if hit is not None:
            accepted.append(rec(i, hit, "model", "related"))
            continue
        near = [int(j) for j in top if sim(i, j) >= NEAR_MISS_SIM]
        if near:
            review.append(rec(i, near[0], "model", "related",
                              f"name similarity {sim(i, near[0]):.2f} < {name_min_sim}"))

    return accepted, review, scorer


def scorer_summary(scorer):
    return {
        "features": META_FEATURES,
        "coef": [round(float(c), 4) for c in scorer.coef_[0]],
        "intercept": round(float(scorer.intercept_[0]), 4),
    }
