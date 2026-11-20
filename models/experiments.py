#!/usr/bin/env python3
"""Compare model designs under the same repeated grouped CV as fit_models.py.

Each design maps a model spec to (features, LinearDesign). Metrics are on
held-out same-creature pairs: MAE relative to the mean baseline (lower is
better), R², 80% interval coverage, and mean interval width relative to the
true value.

Usage:
    .venv/bin/python -m models.experiments [--data real|surrogate] [--fields]
    .venv/bin/python -m models.experiments --nested-selection
    .venv/bin/python -m models.experiments --interactions
    .venv/bin/python -m models.experiments --loss
    .venv/bin/python -m models.experiments --two-stage
    .venv/bin/python -m models.experiments --combat
    .venv/bin/python -m models.experiments --max-features [--seed N --repeats N]
"""

import os

# One BLAS/OpenMP thread per process: the experiments run one process per
# core, and threaded BLAS in each made Huber fits ~200x slower (3 ms -> 0.6 s)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

from models.fit_models import MODEL_SPECS, LinearDesign, fit_spec  # noqa: E402
from pipeline.datasets import SOURCES, load_pairs, resolve_source  # noqa: E402

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


def nested_ratios(pairs, spec, variants, n_repeats=3, seed=2000, design=None,
                  counterparts=None):
    """Honest held-out error of feature-selection procedures: selection is
    re-run inside each outer training fold. variants: {name: (backward
    pool, forward extras, forward threshold)}. counterparts: stats dropped
    when their coefficient is negative (default: the spec's COUNTERPART).
    Returns {name: mean MAE / baseline over repeats} and {name: Counter of
    kept features}."""
    from models.fit_models import (DEFAULT_DESIGN, counterparts_of, extract, feature_value,
                                   grouped_folds, rule_pairs, select_features, source_target)

    design = design or DEFAULT_DESIGN
    pairs = rule_pairs(pairs, spec)
    src, tgt = source_target(spec)
    full = list(dict.fromkeys(f for pool, extras, _ in variants.values() for f in pool + extras))
    X, y, g, same = extract(pairs, src, tgt, full, spec["target"])
    usable = [p for p in pairs if p[tgt].get(spec["target"]) is not None
              and all(feature_value(p, src, f) is not None for f in full)]
    ratios = {n: [] for n in variants}
    kept_count = {n: Counter() for n in variants}
    for rep in range(n_repeats):
        pred = {n: np.empty_like(y) for n in variants}
        base = np.empty_like(y)
        for test in grouped_folds(g, 5, seed + rep):
            tr = np.setdiff1d(np.arange(len(y)), test)
            for n, (pool, extras, tol_add) in variants.items():
                kept, _ = select_features([usable[k] for k in tr], dict(spec, features=pool),
                                          design,
                                          counterparts or counterparts_of(spec),
                                          extras=extras, tol_add=tol_add)
                kept_count[n].update(kept)
                idx = [full.index(f) for f in kept]
                m = design()
                m.features = kept
                pred[n][test] = m.fit(X[tr][:, idx], y[tr], same[tr]).predict(X[test][:, idx])
            base[test] = y[tr].mean()
        b = np.abs(y - base)[same].mean()
        for n in variants:
            ratios[n].append(np.abs(y - pred[n])[same].mean() / b)
    return {n: float(np.mean(r)) for n, r in ratios.items()}, kept_count


def nested_selection(pairs, n_repeats=3):
    """Honest cost of feature pruning: the shipped selection re-run inside
    each outer training fold, vs. the unpruned explainable design. Returns
    rows of (field, unpruned MAE/baseline, pruned MAE/baseline)."""
    from models.fit_models import (DEFAULT_DESIGN, candidate_features, extract, grouped_folds,
                                   rule_pairs, source_target)

    rows = []
    for spec in MODEL_SPECS:
        src, tgt = source_target(spec)
        full = candidate_features(spec)
        sub = rule_pairs(pairs, spec)
        X, y, g, same = extract(sub, src, tgt, full, spec["target"])
        unpruned = []
        for rep in range(n_repeats):
            pred, base = np.empty_like(y), np.empty_like(y)
            for test in grouped_folds(g, 5, 2000 + rep):
                tr = np.setdiff1d(np.arange(len(y)), test)
                m = DEFAULT_DESIGN()
                m.features = full
                pred[test] = m.fit(X[tr], y[tr], same[tr]).predict(X[test])
                base[test] = y[tr].mean()
            unpruned.append(np.abs(y - pred)[same].mean() / np.abs(y - base)[same].mean())
        pruned, _ = nested_ratios(pairs, spec, {"pruned": (full, [], 0.0)}, n_repeats)
        rows.append((f"{spec['direction']}/{spec['target']}", float(np.mean(unpruned)),
                     pruned["pruned"]))
    return rows


def interaction_variants(spec):
    """Candidate pools for role and interaction terms (see --interactions).
    Tier slopes: elite/solo x level/CR (and x the counterpart stat);
    combat roles: 4e artillery/brute/controller/lurker/soldier indicators
    (skirmisher = reference), either in the backward pool or offered
    afterwards one at a time, entering only if they cut CV error by 3%."""
    from models.fit_models import (COMBAT_ROLES, COUNTERPART, ROLE_FEATURES, SCALE,
                                   candidate_features)

    scale = SCALE[spec["direction"]]
    base = spec["features"] + ROLE_FEATURES
    tiers = ("role_elite", "role_solo")  # minions: 6 pairs, none same-creature
    slopes = [f"{t}*{scale}" for t in tiers]
    stat_slopes = [f"{t}*{c}" for t in tiers for c in COUNTERPART[spec["direction"]][spec["target"]]]
    combat = [f"role_{r}" for r in COMBAT_ROLES]
    return {
        "tiers only (previous)": (base, [], 0.0),
        "shipped": (candidate_features(spec), [], 0.0),
        "+ tier slopes on level/CR": (base + slopes, [], 0.0),
        "+ tier slopes on all terms": (base + slopes + stat_slopes, [], 0.0),
        "+ combat roles (backward)": (base + combat, [], 0.0),
        "+ everything (backward)": (base + slopes + stat_slopes + combat, [], 0.0),
        "shipped, + combat (forward 3%)": (candidate_features(spec), combat, 0.03),
    }


LOSS_DESIGNS = {
    "least squares (current)": dict(fit="ols"),
    "least squares, related x0.5": dict(fit="ols", related_weight=0.5),
    "least squares, same-creature only": dict(fit="ols", related_weight=0.0),
    "huber": dict(fit="huber"),
    "huber, related x0.5": dict(fit="huber", related_weight=0.5),
    "huber, related x0.25": dict(fit="huber", related_weight=0.25),
}


def _loss_job(pairs, n_repeats, job):
    from models.fit_models import candidate_features

    spec, name = job
    design = LinearDesign(name, interval="scaled", round_sig=2, **LOSS_DESIGNS[name])
    r, kept = nested_ratios(pairs, spec, {name: (candidate_features(spec), [], 0.0)},
                            n_repeats, design=design)
    return r[name], sum(kept[name].values()) / (5 * n_repeats)


def losses(pairs, n_repeats=3):
    """Nested-CV comparison of the fitting loss and the training weight of
    related pairs, with term selection re-run inside each outer fold."""
    names = list(LOSS_DESIGNS)
    jobs = [(spec, n) for spec in MODEL_SPECS for n in names]
    with ProcessPoolExecutor() as ex:
        out = dict(zip(((f"{s['direction']}/{s['target']}", n) for s, n in jobs),
                       ex.map(partial(_loss_job, pairs, n_repeats), jobs)))
    keys = [f"{s['direction']}/{s['target']}" for s in MODEL_SPECS]
    prev = [out[(k, names[0])][0] for k in keys]
    print(f"{'design':34s} {'median':>7s} {'mean':>6s} {'terms':>6s}  fields better/worse >1%")
    for n in names:
        v = [out[(k, n)][0] for k in keys]
        print(f"{n:34s} {np.median(v):7.3f} {np.mean(v):6.3f} "
              f"{np.mean([out[(k, n)][1] for k in keys]):6.2f}  "
              f"{sum(a < b * 0.99 for a, b in zip(v, prev)):2d}/"
              f"{sum(a > b * 1.01 for a, b in zip(v, prev)):<2d}")
    print(f"\n{'field (MAE / baseline)':26s} " + " ".join(f"{k:>6d}" for k in range(len(names))))
    for k in keys:
        row = [out[(k, n)][0] for n in names]
        best = int(np.argmin(row))
        print(f"{k:26s} " + " ".join(f"{v:5.3f}{'*' if i == best else ' '}"
                                     for i, v in enumerate(row)))


def two_stage(pairs, n_repeats=10):
    """Rules on the source scale (direct) vs. on the target edition's scale,
    with that scale converted by a level/CR rule fit on the training folds
    (two-stage) or given (the GM picks it). Unpruned: counterpart + scale +
    role tiers, Huber, related pairs at half weight. MAE / baseline."""
    from models.fit_models import (COUNTERPART, ROLE_FEATURES, SCALE, TARGET_SCALE,
                                   feature_value, grouped_folds, rule_pairs, source_target)

    def fit(X, y, same):
        m = LinearDesign("x", fit="huber", interval="scaled", related_weight=0.5)
        m.features = [f"f{k}" for k in range(X.shape[1])]
        return m.fit(X, y, same)

    for direction in ("4e_to_5e", "5e_to_4e"):
        print(f"\n== {direction:16s} direct  two-stage  given scale")
        s_scale, t_scale = SCALE[direction], TARGET_SCALE[direction]
        for target, cps in COUNTERPART[direction].items():
            if target == t_scale:
                continue
            spec = {"direction": direction, "target": target}
            src, tgt = source_target(spec)
            rows = [p for p in rule_pairs(pairs, spec)
                    if p[tgt].get(target) is not None and p[tgt].get(t_scale) is not None
                    and all(p[src].get(c) is not None for c in cps + [s_scale])]
            y = np.array([p[tgt][target] for p in rows], float)
            g = np.array([p["group"] for p in rows])
            same = np.array([p["match_type"] == "same" for p in rows])
            tiers = np.array([[feature_value(p, src, t) for t in ROLE_FEATURES] for p in rows])
            C = np.array([[p[src][c] for c in cps] for p in rows]).reshape(len(rows), -1)
            S = np.array([[p[src][s_scale]] for p in rows], float)
            T = np.array([[p[tgt][t_scale]] for p in rows], float)
            ratios = {k: [] for k in ("direct", "two-stage", "given")}
            for rep in range(n_repeats):
                P = {k: np.empty_like(y) for k in ratios}
                base = np.empty_like(y)
                for test in grouped_folds(g, 5, 1000 + rep):
                    tr = np.setdiff1d(np.arange(len(y)), test)
                    Xd = np.hstack([C, S, tiers])
                    P["direct"][test] = fit(Xd[tr], y[tr], same[tr]).predict(Xd[test])
                    Xs = np.hstack([S, tiers])
                    T_hat = fit(Xs[tr], T[tr, 0], same[tr]).predict(Xs)[:, None]
                    Xt = np.hstack([C, T, tiers])
                    mt = fit(Xt[tr], y[tr], same[tr])
                    P["given"][test] = mt.predict(Xt[test])
                    P["two-stage"][test] = mt.predict(np.hstack([C, T_hat, tiers])[test])
                    base[test] = y[tr].mean()
                b = np.abs(y - base)[same].mean()
                for k in ratios:
                    ratios[k].append(np.abs(y - P[k])[same].mean() / b)
            v = [np.mean(ratios[k]) for k in ratios]
            print(f"  {target:18s} {v[0]:6.3f}  {v[1]:9.3f}  {v[2]:11.3f}")


def _combat_job(pairs, n_repeats, job):
    from models.fit_models import COMBAT_ROLES, candidate_features, direct_spec

    spec, mode = job
    if mode == "direct":
        spec = direct_spec(spec)
    pool = candidate_features(spec)
    combat = [f"role_{r}" for r in COMBAT_ROLES]
    return nested_ratios(pairs, spec, {"tiers": (pool, [], 0.0),
                                       "+ combat (forward 3%)": (pool, combat, 0.03)},
                         n_repeats)


def combat_roles(pairs, n_repeats=5):
    """Nested CV: 4e combat roles (artillery, brute, controller, lurker,
    soldier vs. skirmisher) offered to each rule's forward step, entering
    only if they cut CV error by 3%. Rules on the target level/CR and the
    direct rules, separately."""
    from models.fit_models import TARGET_SCALE

    jobs = [(s, mode) for s in MODEL_SPECS for mode in ("target", "direct")
            if mode == "target" or s["target"] != TARGET_SCALE[s["direction"]]]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(partial(_combat_job, pairs, n_repeats), jobs))
    for mode in ("target", "direct"):
        rows = [(s, r) for (s, m), r in zip(jobs, res) if m == mode]
        a = [r[0]["tiers"] for _, r in rows]
        b = [r[0]["+ combat (forward 3%)"] for _, r in rows]
        print(f"\n== rules on the {mode} level/CR: median {np.median(a):.3f} -> {np.median(b):.3f}, "
              f"mean {np.mean(a):.3f} -> {np.mean(b):.3f}, better/worse >1%: "
              f"{sum(y < x * 0.99 for x, y in zip(a, b))}/{sum(y > x * 1.01 for x, y in zip(a, b))}")
        for (s, (r, kept)), x, y in zip(rows, a, b):
            roles = [(f[5:], c) for f, c in kept["+ combat (forward 3%)"].most_common()
                     if f[5:] in ("artillery", "brute", "controller", "lurker", "soldier")]
            print(f"  {s['direction'] + '/' + s['target']:26s} {x:.3f} -> {y:.3f} "
                  f"({(y - x) / x:+5.1%})  entered: {roles}")


# "The better of two abilities", where an edition's own math uses it: 4e
# defenses take the higher of two abilities (Fort STR/CON, Ref DEX/INT, Will
# WIS/CHA; light-armor AC adds DEX or INT), and 5e weapon attacks and their
# damage add STR or DEX. 4e -> 5e saves split each 4e defense into abilities,
# so the 4e defense pair (and the defense itself) are offered to them.
MAX_PAIRS = {
    ("4e_to_5e", "attack_bonus"): ("ability_str", "ability_dex"),
    ("4e_to_5e", "damage_avg"): ("ability_str", "ability_dex"),
    ("4e_to_5e", "damage_per_round"): ("ability_str", "ability_dex"),
    ("4e_to_5e", "save_dex"): ("ability_dex", "ability_int"),
    ("4e_to_5e", "save_con"): ("ability_str", "ability_con"),
    ("4e_to_5e", "save_wis"): ("ability_wis", "ability_cha"),
    ("4e_to_5e", "save_cha"): ("ability_wis", "ability_cha"),
    ("5e_to_4e", "ac"): ("ability_dex", "ability_int"),
    ("5e_to_4e", "fort"): ("ability_str", "ability_con"),
    ("5e_to_4e", "ref"): ("ability_dex", "ability_int"),
    ("5e_to_4e", "will"): ("ability_wis", "ability_cha"),
    ("5e_to_4e", "attack_bonus"): ("ability_str", "ability_dex"),
    ("5e_to_4e", "damage_avg"): ("ability_str", "ability_dex"),
}
SAVE_DEFENSE = {"save_dex": "ref", "save_con": "fort", "save_wis": "will", "save_cha": "will"}


def max_variants(spec, direct):
    """Candidate pools for one rule (see --max-features), each with the
    shipped forward step (combat roles on the 5e -> 4e target-level rules)."""
    from models.fit_models import COMBAT_FORWARD, COMBAT_ROLES, COUNTERPART, candidate_features

    key = (spec["direction"], spec["target"])
    a, b = MAX_PAIRS[key]
    mx = f"max({a},{b})"
    pool = candidate_features(spec)
    combat = [f"role_{r}" for r in COMBAT_ROLES] if key in COMBAT_FORWARD and not direct else []
    tol = 0.03 if combat else 0.0
    cps = COUNTERPART[spec["direction"]][spec["target"]]
    without = [f for f in pool if f != mx]  # the pool before MAX_TERMS shipped
    variants = {
        "no max": (without, combat, tol),
        "+ max": (without + [mx], combat, tol),
        "+ both abilities": (without + [f for f in (a, b) if f not in without], combat, tol),
    }
    if a in cps or b in cps:
        variants["max replaces abilities"] = ([mx] + [f for f in without if f not in (a, b)],
                                              combat, tol)
    if spec["target"] in SAVE_DEFENSE:
        variants["+ 4e defense"] = (without + [SAVE_DEFENSE[spec["target"]]], combat, tol)
    return variants, cps + [mx, a, b] + [SAVE_DEFENSE.get(spec["target"], mx)]


def _max_job(pairs, n_repeats, seed, job):
    """One (rule, mode, candidate pool): same outer folds for every pool."""
    from models.fit_models import direct_spec

    spec, mode, name = job
    if mode == "direct":
        spec = direct_spec(spec)
    variants, cps = max_variants(spec, mode == "direct")
    r, kept = nested_ratios(pairs, spec, {name: variants[name]}, n_repeats, seed=seed,
                            counterparts=cps)
    return r[name], kept[name]


def max_features(pairs, n_repeats=5, seed=2000):
    """Nested CV: max(a, b) of two abilities offered to the rules whose
    edition math uses the better of two (MAX_PAIRS), vs. offering both
    abilities separately, max replacing the abilities, and (4e -> 5e saves)
    the 4e defense. Rules on the target level/CR and direct rules. Each
    rule's line prints as soon as all its pools are scored."""
    from models.fit_models import TARGET_SCALE, direct_spec

    rules = [(s, mode) for s in MODEL_SPECS for mode in ("target", "direct")
             if (s["direction"], s["target"]) in MAX_PAIRS
             and (mode == "target" or s["target"] != TARGET_SCALE[s["direction"]])]
    names = {k: list(max_variants(direct_spec(s) if m == "direct" else s, m == "direct")[0])
             for k, (s, m) in enumerate(rules)}
    print(f"MAE / baseline (change vs. no max); [max: folds keeping the max term, "
          f"of {5 * n_repeats}]", flush=True)
    done = {k: {} for k in names}
    with ProcessPoolExecutor() as ex:
        futs = {ex.submit(_max_job, pairs, n_repeats, seed, (*rules[k], n)): (k, n)
                for k in names for n in names[k]}
        for fut in as_completed(futs):
            k, n = futs[fut]
            done[k][n] = fut.result()
            if len(done[k]) < len(names[k]):
                continue
            s, mode = rules[k]
            base = done[k]["no max"][0]
            cells = [f"no max {base:.3f}"] + [
                f"{n}: {v:.3f} ({(v - base) / base:+.1%})"
                for n, (v, _) in ((n, done[k][n]) for n in names[k][1:])]
            kept = done[k]["+ max"][1]
            n_max = sum(c for f, c in kept.items() if f.startswith("max("))
            print(f"{mode:6s} {s['direction'] + '/' + s['target']:24s} " + " | ".join(cells)
                  + f"  [max: {n_max}]", flush=True)


def _interaction_job(pairs, n_repeats, spec):
    return nested_ratios(pairs, spec, interaction_variants(spec), n_repeats)


def interactions(pairs, n_repeats=3):
    """Nested-CV comparison of role/interaction candidate pools, per field."""
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(partial(_interaction_job, pairs, n_repeats), MODEL_SPECS))
    names = list(interaction_variants(MODEL_SPECS[0]))
    prev = [r[names[0]] for r, _ in res]
    print(f"{'candidate pool':32s} {'median':>7s} {'mean':>6s}  fields better/worse >1%")
    for n in names:
        v = [r[n] for r, _ in res]
        print(f"{n:32s} {np.median(v):7.3f} {np.mean(v):6.3f}  "
              f"{sum(a < b * 0.99 for a, b in zip(v, prev)):2d}/"
              f"{sum(a > b * 1.01 for a, b in zip(v, prev)):<2d}")
    print(f"\n{'field (MAE / baseline)':26s} " + " ".join(f"{k:>6d}" for k in range(len(names)))
          + "   most-kept role terms, 'everything' pool (of 15 folds)")
    for spec, (r, kept) in zip(MODEL_SPECS, res):
        row = [r[n] for n in names]
        best = int(np.argmin(row))
        extra = [(f, c) for f, c in kept["+ everything (backward)"].most_common()
                 if f.startswith("role_") or "*" in f][:3]
        print(f"{spec['direction'] + '/' + spec['target']:26s} "
              + " ".join(f"{v:5.3f}{'*' if k == best else ' '}" for k, v in enumerate(row))
              + "   " + ", ".join(f"{f}:{c}" for f, c in extra))


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
    ap.add_argument("--interactions", action="store_true",
                    help="role/interaction candidate pools under nested CV")
    ap.add_argument("--loss", action="store_true",
                    help="fitting loss and related-pair weight under nested CV")
    ap.add_argument("--combat", action="store_true",
                    help="4e combat roles offered to each rule's forward step, nested CV")
    ap.add_argument("--max-features", action="store_true",
                    help="max(a, b) of two abilities offered to the rules that use one, nested CV")
    ap.add_argument("--seed", type=int, default=2000, help="outer-fold seed (--max-features)")
    ap.add_argument("--repeats", type=int, default=5, help="outer repeats (--max-features)")
    ap.add_argument("--two-stage", action="store_true",
                    help="rules on the target edition's level/CR, given or converted")
    args = ap.parse_args(argv)
    pairs = load_pairs(resolve_source(args.data))

    if args.interactions:
        interactions(pairs)
        return
    if args.loss:
        losses(pairs)
        return
    if args.two_stage:
        two_stage(pairs)
        return
    if args.combat:
        combat_roles(pairs)
        return
    if args.max_features:
        max_features(pairs, args.repeats, args.seed)
        return

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
