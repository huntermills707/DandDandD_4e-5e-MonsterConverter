#!/usr/bin/env python3
"""Fit per-field linear regressions for both 4e -> 5e and 5e -> 4e.

Evaluation is repeated grouped K-fold cross-validation. Pairs are grouped by
connected monsters (pipeline/build_pairs.assign_groups), so variants of one
5e monster never sit on both sides of a split. Metrics are computed on
held-out same-creature pairs (the ones with a real ground truth) and
reported as mean and standard deviation over repeats.

Prediction intervals are cross-conformal and scale-adaptive: the relative
residual quantile q gives [pred * (1 - q), pred * (1 + q)]. Coverage is
estimated honestly with nested CV (q chosen on the training folds only).
The shipped equation and q are fit on all pairs.

Results go to models/models.json (for the app) and models/eval_report.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.datasets import SOURCES, load_pairs, resolve_source  # noqa: E402

MODELS_DIR = ROOT / "models"

ALPHA = 0.20  # nominal miscoverage; 80% scale-adaptive prediction intervals
N_FOLDS = 5
N_REPEATS = 10
MIN_N = 15
ADD_TOL = 0.03  # a role/interaction term must cut CV error by 3% to enter

# Each spec: direction, target field on the target side, and features: the
# target's counterpart stat(s) in the other edition plus the TARGET edition's
# scale (5e CR for 4e -> 5e, 4e level for 5e -> 4e). The scale rule itself
# (CR from level, level from CR) uses the source scale. Rules on the target
# scale are a GM's own monster math ("4e AC ~ level + 14"): the GM picks the
# level/CR, or the scale rule converts it (two-stage). With the true target
# level, 5e -> 4e AC/defense/attack error halves; with the converted one it
# matches rules on the source scale (models/experiments.py --two-stage).
# Simpler specs with more features fit no better under grouped CV.
SCALE = {"4e_to_5e": "level", "5e_to_4e": "cr_num"}          # source-side scale
TARGET_SCALE = {"4e_to_5e": "cr_num", "5e_to_4e": "level"}   # target-side scale
OTHER_EDITION = {"4e": "5e", "5e": "4e"}
TARGET_SCALE_OF = {"4e": "cr_num", "5e": "level"}  # source edition -> target scale
ABILITIES = ("str", "dex", "con", "int", "wis", "cha")
COUNTERPART = {
    "4e_to_5e": {
        "cr_num": [], "hp": ["hp"], "ac": ["ac"], "attack_bonus": ["attack_bonus"],
        # 5e damage: best single attack (riders included) and the Multiattack
        # routine per round; 4e damage is the basic attack, and a 4e
        # "makes two attacks" power suggests a 5e Multiattack
        "damage_avg": ["damage_avg"],
        "damage_per_round": ["damage_avg", "attacks_per_round"],
        **{f"ability_{a}": [f"ability_{a}"] for a in ABILITIES},
        # 5e lists only proficient saves: modifier + proficiency
        **{f"save_{a}": [f"ability_{a}"] for a in ("dex", "con", "wis", "cha")},
    },
    "5e_to_4e": {
        "level": [], "hp": ["hp"], "ac": ["ac"],
        # 4e defenses use the better of two abilities
        "fort": ["ability_str", "ability_con"], "ref": ["ability_dex"],
        "will": ["ability_wis", "ability_cha"],
        "attack_bonus": ["attack_bonus"],
        # 4e basic attack damage from 5e per-attack and per-round damage
        "damage_avg": ["damage_avg", "damage_per_round"],
        **{f"ability_{a}": [f"ability_{a}"] for a in ABILITIES},
    },
}
MODEL_SPECS = [
    {"direction": d, "target": t,
     "features": feats + [SCALE[d] if t == TARGET_SCALE[d] else TARGET_SCALE[d]]}
    for d, targets in COUNTERPART.items() for t, feats in targets.items()
]


def direct_spec(spec):
    """The same rule on the source scale (5e CR for 5e -> 4e): used when the
    GM leaves the target level/CR to be converted. Rules on a converted
    level/CR do no better, and on the 5e saves worse (selection, scored with
    the true CR, drops the 4e ability), so both rules are kept."""
    d = spec["direction"]
    return dict(spec, features=COUNTERPART[d][spec["target"]] + [SCALE[d]])


ROLE_TIERS = ("minion", "elite", "solo")  # 4e role tiers; "standard" is the reference
# 4e combat roles; skirmisher is the reference (the DMG's baseline monster:
# AC level + 14, 8 HP per level). Minions carry no combat role in the data.
COMBAT_ROLES = ("artillery", "brute", "controller", "lurker", "soldier")


def role_tier(role):
    r = (role or "").lower()
    return next((t for t in ROLE_TIERS if t in r), "standard")


def role_flags(role):
    """Set of 4e role words present: tier, combat role, and "leader"."""
    r = (role or "").lower()
    return {w for w in ROLE_TIERS + COMBAT_ROLES + ("skirmisher", "leader") if w in r}


def weighted_median(x, w):
    """Smallest x with at least half the total weight at or below it."""
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(np.asarray(w, dtype=float)[order])
    return float(np.asarray(x)[order][np.searchsorted(cum, cum[-1] / 2)])


class LinearDesign:
    """A linear model in (optionally) transformed space.

        z    = intercept + sum(coef_k * t_k(x_k))     t_k = log1p or identity
        pred = inverse_target(z)                       log -> expm1(z)

    fit:      "nonneg" (coefficients and intercept >= 0), "ols", "ridge"
              (standardized features, penalty by efficient leave-one-out;
              coefficients reported back in original units), or "huber"
              (squared loss for small residuals, absolute for large ones:
              a few odd monsters can't drag the rule; epsilon 1.35)
    related_weight: training weight of related (not same-creature) pairs;
              fit(X, y, same) takes the same-creature mask
    interval: "relative" [pred(1-q), pred(1+q)], "absolute" [pred-q, pred+q],
              "log" [expm1(z-q), expm1(z+q)] (multiplicative), or "scaled"
              pred +- q(|pred| + c) with c = median |target| in training:
              absolute-like for small values, relative-like for large ones
    """

    def __init__(self, name="nonneg_linear", fit="nonneg", interval="relative",
                 log_target=False, log_features=(), clip_min=0.0, round_sig=None,
                 related_weight=1.0):
        self.name, self.fit_kind, self.interval = name, fit, interval
        self.round_sig, self.related_weight = round_sig, related_weight
        self.log_target, self.log_features, self.clip_min = log_target, set(log_features), clip_min
        self.features = None

    def __call__(self):  # acts as its own factory: make_model() -> fresh copy
        d = LinearDesign(self.name, self.fit_kind, self.interval, self.log_target,
                         self.log_features, self.clip_min, self.round_sig, self.related_weight)
        d.features = self.features
        return d

    def _tx(self, X):
        X = np.array(X, dtype=float)
        for k, f in enumerate(self.features):
            if f in self.log_features:
                X[:, k] = np.log1p(np.maximum(X[:, k], 0))
        return X

    def fit(self, X, y, same=None):
        """same: same-creature mask (None = all); related pairs get
        `related_weight`, and weight 0 leaves them out."""
        self.scale_offset_ = float(np.median(np.abs(y)))
        if self.round_sig:
            self.scale_offset_ = float(f"{self.scale_offset_:.2g}")
        w = np.ones(len(y)) if same is None else np.where(same, 1.0, self.related_weight)
        keep = w > 0
        X, y, w = np.asarray(X)[keep], np.asarray(y)[keep], w[keep]
        Xt = self._tx(X)
        yt = np.log1p(np.maximum(y, 0)) if self.log_target else y
        robust = self.fit_kind == "huber"
        if Xt.shape[1] == 0:  # intercept-only rule: the typical value
            center = weighted_median(yt, w) if robust else np.average(yt, weights=w)
            self.intercept_, self.coef_ = float(center), np.zeros(0)
        elif robust:
            sd = Xt.std(axis=0)
            sd[sd == 0] = 1.0
            m = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000)
            m.fit(Xt / sd, yt, sample_weight=w)
            self.intercept_, self.coef_ = float(m.intercept_), m.coef_ / sd
        elif self.fit_kind == "nonneg":
            m = LinearRegression(fit_intercept=False, positive=True)
            m.fit(np.column_stack([np.ones(len(Xt)), Xt]), yt)
            self.intercept_, self.coef_ = float(m.coef_[0]), m.coef_[1:]
        elif self.fit_kind == "ridge":
            mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
            sd[sd == 0] = 1.0
            m = RidgeCV(alphas=np.logspace(-2, 3, 26)).fit((Xt - mu) / sd, yt)
            self.coef_ = m.coef_ / sd
            self.intercept_ = float(m.intercept_ - (mu / sd) @ m.coef_)
            self.alpha_ = float(m.alpha_)
        else:
            m = LinearRegression().fit(Xt, yt, sample_weight=w)
            self.intercept_, self.coef_ = float(m.intercept_), m.coef_
        if self.round_sig:
            # hand-usable equation: coefficients to `round_sig` significant
            # figures, then the intercept refit for them (weighted median
            # for huber, mean otherwise) and rounded to 0.1
            self.coef_ = np.array([float(f"{c:.{self.round_sig}g}") for c in self.coef_])
            r = yt - Xt @ self.coef_
            center = weighted_median(r, w) if robust else np.average(r, weights=w)
            self.intercept_ = round(float(center), 1)
        return self

    def link(self, X):
        return self.intercept_ + self._tx(X) @ self.coef_

    def predict(self, X):
        z = self.link(X)
        pred = np.expm1(z) if self.log_target else z
        return np.maximum(pred, self.clip_min)

    # interval calibration: residual on the scale where the interval is symmetric
    def residual(self, y, pred):
        if self.interval == "relative":
            return np.abs(y - pred) / np.maximum(np.abs(pred), 1e-6)
        if self.interval == "log":
            return np.abs(np.log1p(np.maximum(y, 0)) - np.log1p(pred))
        if self.interval == "scaled":
            return np.abs(y - pred) / (np.abs(pred) + self.scale_offset_)
        return np.abs(y - pred)

    def bounds(self, pred, q):
        if self.interval == "relative":
            lo, hi = pred * (1 - q), pred * (1 + q)
        elif self.interval == "log":
            lo, hi = np.expm1(np.log1p(pred) - q), np.expm1(np.log1p(pred) + q)
        elif self.interval == "scaled":
            half = q * (np.abs(pred) + self.scale_offset_)
            lo, hi = pred - half, pred + half
        else:
            lo, hi = pred - q, pred + q
        return np.maximum(lo, self.clip_min), hi

    def describe(self):
        return {"estimator": self.name, "fit": self.fit_kind, "interval": self.interval,
                "round_sig": self.round_sig, "related_weight": self.related_weight,
                "log_target": self.log_target,
                "log_features": sorted(self.log_features & set(self.features)),
                "clip_min": self.clip_min,
                "interval_offset": round(getattr(self, "scale_offset_", 0.0), 4)}


NonNegLinear = LinearDesign()  # the original design

# Chosen design (models/experiments.py, D9), picked for being easy to apply
# by hand at no measurable cost in accuracy: counterpart stat + level/CR +
# 4e role adjustments, coefficients rounded to 2 significant figures,
# interval  result +- q * (result + c). Fit with Huber loss, related pairs at
# half weight (experiments.py --loss, nested CV: median error 60% -> 55% of
# baseline, 23 fields better, 2 worse): a few odd monsters no longer drag a
# rule, and variants count less than the same creature.
ROLE_FEATURES = ["role_minion", "role_elite", "role_solo"]
DEFAULT_DESIGN = LinearDesign("explainable", fit="huber", interval="scaled", round_sig=2,
                              related_weight=0.5)

# Role interactions offered to the backward selection, by (direction, target).
# A slope replaces its tier's flat bonus. 4e multiplies solo HP by 4, so the
# solo bonus on 4e HP is proportional to level: a slope on 4e level, zero at
# level 0. (On the CR scale, slope plus flat bonus fit a little better but
# went negative below CR 5, outside the data's solos.) An elite slope was never selected; 4e's
# other role adjustments are flat (+2 to defenses and so on), and neither
# tier slopes elsewhere nor combat roles (brute, lurker, ...) improved
# held-out error (models/experiments.py --interactions), so they are not
# offered.
INTERACTIONS = {("5e_to_4e", "hp"): ["role_solo*{scale}"]}  # the rule's level/CR

# 4e combat roles (artillery, brute, controller, lurker, soldier vs.
# skirmisher) are offered to a rule's forward step (entering only if they cut
# CV error by 3%) where 4e's own role math applies: the stats the 4e DMG
# adjusts by role (HP, AC, defenses, attack, damage), converting TO 4e, on
# the target level, where the role is the GM's choice for the converted
# monster. Nested CV (models/experiments.py --combat): AC -28%, attack -20%,
# HP and Ref -8%, damage -4% (brute and soldier enter in nearly every fold).
# Offered to 4e -> 5e rules or the direct rules, they mostly chased noise
# (4e -> 5e saves 2-8% worse), so they are not offered there.
COMBAT_FORWARD = {("5e_to_4e", t) for t in
                  ("hp", "ac", "fort", "ref", "will", "attack_bonus", "damage_avg")}

# Stats 4e fixes by role tier rather than scale: every minion has 1 HP. Those
# pairs are left out of the rule's fit, and the rule returns the fixed value.
FIXED_BY_ROLE = {("5e_to_4e", "hp"): {"minion": 1.0}}


def source_target(spec):
    if spec["direction"] == "4e_to_5e":
        return "4e", "5e"
    return "5e", "4e"


def rule_pairs(pairs, spec):
    """Pairs a rule is fit on: those whose 4e role tier doesn't fix the target."""
    fixed = FIXED_BY_ROLE.get((spec["direction"], spec["target"]), {})
    return [p for p in pairs if role_tier(p["4e"].get("role")) not in fixed]


def candidate_features(spec, role_features=ROLE_FEATURES):
    """Counterpart + scale, role tiers (minus fixed ones and those replaced
    by a slope), and interactions."""
    key = (spec["direction"], spec["target"])
    fixed = FIXED_BY_ROLE.get(key, {})
    scale = next(f for f in spec["features"] if f in ("level", "cr_num"))
    slopes = [f.format(scale=scale) for f in INTERACTIONS.get(key, [])]
    sloped = {f.split("*")[0] for f in slopes}
    return (spec["features"]
            + [f for f in role_features if f[5:] not in fixed and f not in sloped] + slopes)


def feature_value(pair, source_prefix, feature):
    """Source-side stat, the target edition's level/CR, a 4e role indicator
    ("role_elite", "role_brute", "role_leader"), or a product of two of
    those ("role_solo*level"). Role
    comes from the 4e side in both directions: for 5e -> 4e it is the role
    the converted monster should have."""
    if "*" in feature:
        a, b = (feature_value(pair, source_prefix, f) for f in feature.split("*"))
        return None if a is None or b is None else a * b
    if feature.startswith("role_"):
        return float(feature[5:] in role_flags(pair["4e"].get("role")))
    if feature == TARGET_SCALE_OF[source_prefix]:
        # the target edition's level/CR: chosen by the GM, or converted
        return pair[OTHER_EDITION[source_prefix]].get(feature)
    return pair[source_prefix].get(feature)


def extract(pairs, source_prefix, target_prefix, features, target):
    """Rows with the target and every feature present.
    Returns X, y, groups, is_same (same-creature pair)."""
    X, y, groups, same = [], [], [], []
    for k, p in enumerate(pairs):
        tgt = p[target_prefix]
        row = [feature_value(p, source_prefix, f) for f in features]
        if tgt.get(target) is None or any(v is None for v in row):
            continue
        X.append(row)
        y.append(tgt[target])
        groups.append(p.get("group", k))
        same.append(p.get("match_type", "same") == "same")
    X = np.array(X, dtype=float) if features else np.zeros((len(y), 0))
    return (X.reshape(len(y), len(features)), np.array(y, dtype=float),
            np.array(groups), np.array(same, dtype=bool))


def grouped_folds(groups, n_folds, seed):
    """Test-index arrays: unique groups shuffled and dealt into folds."""
    uniq = np.unique(groups)
    order = np.random.default_rng(seed).permutation(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(order)}
    folds = np.array([fold_of[g] for g in groups])
    return [np.flatnonzero(folds == f) for f in range(n_folds)]


def conformal_q(rel_residuals, alpha):
    """Conservative conformal quantile of relative residuals."""
    n = len(rel_residuals)
    if n == 0:
        return 0.0
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(rel_residuals, level, method="higher"))


def oof_predictions(X, y, groups, make_model, n_folds, seed, is_same=None):
    pred = np.empty_like(y)
    same = np.ones(len(y), dtype=bool) if is_same is None else is_same
    for test in grouped_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = make_model().fit(X[train], y[train], same[train]).predict(X[test])
    return pred


def _score(per, y, pred, base, lo, hi, m):
    """Append one repeat's metrics (same-creature rows m) to per."""
    yt, pt = y[m], pred[m]
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    per["r2"].append(1 - ((yt - pt) ** 2).sum() / ss_tot if ss_tot else np.nan)
    per["mae"].append(np.abs(yt - pt).mean())
    per["baseline_mae"].append(np.abs(yt - base[m]).mean())
    covered = (yt >= lo[m]) & (yt <= hi[m])
    per["coverage"].append(covered.mean())
    # conditional coverage by thirds of the prediction
    band = np.digitize(pt, np.quantile(pt, [1 / 3, 2 / 3]))
    for b, name in enumerate(("cov_low", "cov_mid", "cov_high")):
        per[name].append(covered[band == b].mean() if (band == b).any() else np.nan)
    per["rel_width"].append(np.mean((hi[m] - lo[m]) / np.maximum(np.abs(yt), 1e-6)))


def cross_validate(X, y, groups, is_same, make_model, alpha=ALPHA,
                   n_folds=N_FOLDS, n_repeats=N_REPEATS):
    """Repeated grouped CV. Returns per-repeat metric arrays and the
    out-of-fold predictions of every repeat (all pairs), used to calibrate
    the final model's interval."""
    per = {k: [] for k in ("r2", "mae", "baseline_mae", "coverage", "rel_width",
                           "cov_low", "cov_mid", "cov_high")}
    oof_preds = []
    for rep in range(n_repeats):
        seed = 1000 + rep
        pred, base, lo, hi = (np.empty_like(y) for _ in range(4))
        for f, test in enumerate(grouped_folds(groups, n_folds, seed)):
            train = np.setdiff1d(np.arange(len(y)), test)
            model = make_model().fit(X[train], y[train], is_same[train])
            pred[test] = model.predict(X[test])
            base[test] = y[train].mean()
            # nested: interval width chosen on the training folds only
            inner = oof_predictions(X[train], y[train], groups[train], make_model,
                                    n_folds, seed * 31 + f, is_same[train])
            q = conformal_q(model.residual(y[train], inner), alpha)
            lo[test], hi[test] = model.bounds(pred[test], q)
        oof_preds.append(pred)
        _score(per, y, pred, base, lo, hi, is_same)
    return {k: np.array(v) for k, v in per.items()}, np.array(oof_preds)


def cv_mae(X, y, groups, is_same, make_model, n_repeats=3, n_folds=N_FOLDS, seed=500):
    """Mean held-out MAE on same-creature pairs (light CV for selection)."""
    out = []
    for rep in range(n_repeats):
        pred = oof_predictions(X, y, groups, make_model, n_folds, seed + rep, is_same)
        out.append(np.abs(y - pred)[is_same].mean())
    return float(np.mean(out))


def select_features(pairs, spec, make_model, counterparts, extras=(), tol=0.01,
                    tol_add=ADD_TOL):
    """Backward elimination toward short, sensible rules, then a strict
    forward step for role and interaction terms.

    1. A counterpart stat with a negative coefficient is dropped (a stat
       shouldn't convert in the opposite direction; it means the scale
       term is doing the work).
    2. Then, repeatedly, the term whose removal raises CV error least is
       dropped while that rise stays under `tol` (1%).
    3. Then, repeatedly, the candidate in `extras` that lowers CV error most
       is added while it lowers it by more than `tol_add` (4e combat roles,
       for the rules in COMBAT_FORWARD).
    Returns (kept features, dropped features).
    """
    src, tgt = source_target(spec)
    features = list(spec["features"])
    dropped = []

    def data(feats):
        X, y, g, same = extract(pairs, src, tgt, feats, spec["target"])
        model = make_model()
        model.features = feats
        return X, y, g, same, model

    def score(feats):
        X, y, g, same, model = data(feats)
        return cv_mae(X, y, g, same, model)

    while True:
        X, y, g, same, model = data(features)
        model.fit(X, y, same)
        neg = [f for f, c in zip(features, model.coef_) if f in counterparts and c < 0]
        if not neg:
            break
        features.remove(neg[0])
        dropped.append(neg[0])

    current = score(features)
    while features:
        best_mae, best_f = min((score([x for x in features if x != f]), f) for f in features)
        if best_mae > current * (1 + tol):
            break
        features.remove(best_f)
        dropped.append(best_f)
        current = best_mae

    pool = [f for f in extras if f not in features]
    while pool:
        best_mae, best_f = min((score(features + [f]), f) for f in pool)
        if best_mae >= current * (1 - tol_add):
            break
        features.append(best_f)
        pool.remove(best_f)
        current = best_mae
    return features, dropped


def fit_spec(pairs, spec, make_model=NonNegLinear, alpha=ALPHA,
             n_folds=N_FOLDS, n_repeats=N_REPEATS):
    source_prefix, target_prefix = source_target(spec)
    features, target = spec["features"], spec["target"]
    make_model = make_model()
    make_model.features = features
    X, y, groups, is_same = extract(pairs, source_prefix, target_prefix, features, target)
    if len(y) < MIN_N or is_same.sum() < N_FOLDS or len(np.unique(groups)) < N_FOLDS:
        return None

    per, oof_preds = cross_validate(X, y, groups, is_same, make_model, alpha, n_folds, n_repeats)
    final = make_model().fit(X, y, is_same)
    oof_resid = np.concatenate([final.residual(y, pred) for pred in oof_preds])

    def stat(k, nd=3):
        return round(float(np.nanmean(per[k])), nd), round(float(np.nanstd(per[k])), nd)

    r2, r2_sd = stat("r2")
    mae, mae_sd = stat("mae")
    base, base_sd = stat("baseline_mae")
    cov, cov_sd = stat("coverage")
    return {
        "direction": spec["direction"],
        "source_edition": source_prefix,
        "target_edition": target_prefix,
        "target": target,
        "features": features,
        **final.describe(),
        "n": int(len(y)),
        "n_eval": int(is_same.sum()),
        "n_groups": int(len(np.unique(groups))),
        "cv": f"grouped {n_folds}-fold x{n_repeats}, scored on same-creature pairs",
        "intercept": round(float(final.intercept_), 4),
        "coef": [round(float(c), 6) for c in final.coef_],
        "alpha": alpha,
        # rounded up: a slightly wider interval keeps coverage conservative
        "conformal_quantile": math.ceil(conformal_q(oof_resid, alpha) * 100) / 100,
        "r2": r2, "r2_sd": r2_sd,
        "mae": mae, "mae_sd": mae_sd,
        "baseline_mae": base, "baseline_mae_sd": base_sd,
        "test_coverage": cov, "test_coverage_sd": cov_sd,
        "coverage_by_third": [stat(k, 2)[0] for k in ("cov_low", "cov_mid", "cov_high")],
        "interval_rel_width": round(float(np.mean(per["rel_width"])), 3),
    }


LABELS = {
    "cr_num": "CR", "level": "level", "hp": "HP", "ac": "AC", "attack_bonus": "attack",
    "damage_avg": "damage", "damage_per_round": "damage/round",
    "attacks_per_round": "attacks/round", "fort": "Fort", "ref": "Ref", "will": "Will",
    **{f"ability_{a}": a.upper() for a in ABILITIES},
    **{f"save_{a}": f"{a.upper()} save" for a in ABILITIES},
}


def short_label(field):
    if field.startswith("role_"):
        return f"[{field[5:]}]"
    return LABELS.get(field, field)


def num(x):
    """Compact number: 2 significant figures, no trailing zeros."""
    return f"{x:.2g}" if abs(x) < 100 else f"{x:.0f}"


def equation_string(model):
    """e.g. 'HP = 0.87·HP + 39  (elite +140, solo +26·CR; minion = 1)'.
    Role adjustments are grouped by role: a flat bonus, a slope
    ("role_solo*cr_num" -> "solo +26·CR"), or a fixed value."""
    terms, roles = [], {}

    def term(c, label, lead):
        coef = "" if label and abs(abs(c) - 1) < 0.005 else num(abs(c))
        body = f"{coef}·{label}" if coef and label else coef or label
        return f"{'-' if c < 0 else '+'}{lead}{body}"

    for c, f in zip(model["coef"], model["features"]):
        parts = f.split("*")
        role = next((x[5:] for x in parts if x.startswith("role_")), None)
        stats = [x for x in parts if not x.startswith("role_")]
        if role:
            label = "·".join(short_label(x) for x in stats)
            roles.setdefault(role, []).append(term(c, label, ""))
            continue
        label = short_label(f)
        if f in set(model.get("log_features", [])):
            label = f"ln(1+{label})"
        terms.append(term(c, label, " "))
    inter = model["intercept"]
    if inter or not terms:
        terms.append(f"{'-' if inter < 0 else '+'} {abs(inter):g}")
    rhs = " ".join(terms)
    rhs = rhs[2:] if rhs.startswith("+ ") else "-" + rhs[2:]
    eq = f"{short_label(model['target'])} = {rhs}"
    notes = ", ".join(f"{r} {' '.join(t)}" for r, t in roles.items())
    fixed = "; ".join(f"{r} = {v:g}" for r, v in model.get("fixed_by_role", {}).items())
    notes = "; ".join(x for x in (notes, fixed) if x)
    if notes:
        eq += f"  ({notes})"
    return eq


def interval_string(model):
    q, c = model["conformal_quantile"], model.get("interval_offset", 0)
    if model.get("interval") == "scaled":
        return f"± {q:g}·(result + {c:g})"
    if model.get("interval") == "absolute":
        return f"± {q:g}"
    return f"± {q:g}·result"


def write_rules(models, path):
    """Plain-text conversion sheet for use without the app."""
    lines = [
        "# Monster conversion rules",
        "",
        "Generated by `models/fit_models.py`. Each rule converts one stat; the",
        "role terms apply to the 4e role (for 5e → 4e, the role the converted",
        "monster should play; standard = no adjustment). The 80% range is the",
        "interval the true value fell in about 80% of the time under grouped",
        "cross-validation. *Error* is the typical miss relative to always",
        "guessing the average (lower is better).",
        "",
        "**Pick the target level/CR, then convert.** In each direction the first",
        "rule converts the level/CR; choose it yourself if you prefer. The main",
        "table's rules use that *target* level/CR (5e CR in 4e → 5e, 4e level in",
        "5e → 4e), the way each edition's own monster math does. If you'd rather",
        "not settle a level/CR, use the *direct* rules underneath, which work",
        "from the source monster's own level/CR.",
        "",
        "Damage: 4e damage is the basic attack's average; 5e damage is the best",
        "single attack with riders (\"plus 7 (2d6) fire\"), and damage/round",
        "the whole Multiattack routine.",
        "",
    ]

    def row(m, target):
        rule = m["equation"].split(" = ", 1)[1]
        return (f"| {short_label(target)} | {rule} | {interval_string(m)} | "
                f"{m['mae'] / m['baseline_mae']:.0%} | {m['r2']:.2f} |")

    head = ["| Stat | Rule | 80% range | Error | R² |", "|---|---|---|---|---|"]
    for direction, title in (("4e_to_5e", "4e → 5e"), ("5e_to_4e", "5e → 4e")):
        scale = "CR" if TARGET_SCALE[direction] == "cr_num" else "level"
        ms = [m for m in models.values() if m["direction"] == direction]
        lines += [f"## {title}", "", f"With the target {scale} given:", "", *head]
        lines += [row(m, m["target"]) for m in ms]
        lines += ["", f"Direct, without settling the target {scale}:", "", *head]
        lines += [row(m["direct"], m["target"]) for m in ms if "direct" in m]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_rule(pairs, spec, make_model, extra_features, select, direct=False):
    """Candidate terms -> selection -> fit. Returns the result or None."""
    key = (spec["direction"], spec["target"])
    counterparts = COUNTERPART[spec["direction"]][spec["target"]]
    spec = dict(spec, features=candidate_features(spec, list(extra_features)))
    fit_pairs = rule_pairs(pairs, spec)
    dropped = []
    combat = ([f"role_{r}" for r in COMBAT_ROLES]
              if key in COMBAT_FORWARD and not direct else [])
    if select:
        kept, dropped = select_features(fit_pairs, spec, make_model, counterparts,
                                        extras=combat)
        spec = dict(spec, features=kept)
    result = fit_spec(fit_pairs, spec, make_model)
    if result is None:
        return None
    result["dropped_features"] = dropped
    fixed = FIXED_BY_ROLE.get((spec["direction"], spec["target"]))
    if fixed:
        result["fixed_by_role"] = fixed
    result["equation"] = equation_string(result)
    result["interval_rule"] = interval_string(result)
    return result


def fit_all(pairs, make_model=DEFAULT_DESIGN, extra_features=ROLE_FEATURES, select=True,
            verbose=True):
    """Every rule. A rule on the target edition's level/CR carries, under
    "direct", the same stat's rule on the source scale, for when the GM
    leaves the level/CR to be converted."""
    models = {}
    for spec in MODEL_SPECS:
        key = f"{spec['direction']}/{spec['target']}"
        result = fit_rule(pairs, spec, make_model, extra_features, select)
        if result is None:
            if verbose:
                print(f"  skipping {key}: not enough data")
            continue
        if spec["target"] != TARGET_SCALE[spec["direction"]]:
            result["target_scale"] = TARGET_SCALE[spec["direction"]]
            direct = fit_rule(pairs, direct_spec(spec), make_model, extra_features, select,
                              direct=True)
            if direct is not None:
                result["direct"] = {k: v for k, v in direct.items()
                                    if k not in ("direction", "source_edition",
                                                 "target_edition", "target")}
        models[key] = result
        if verbose:
            line = (f"{key:26s} R²={result['r2']:6.2f}±{result['r2_sd']:.2f}  "
                    f"MAE={result['mae']:7.2f} (base {result['baseline_mae']:7.2f})  "
                    f"cov={result['test_coverage']:.2f}")
            if "direct" in result:
                dr = result["direct"]
                line += (f"  | direct: R²={dr['r2']:5.2f} MAE={dr['mae']:7.2f} "
                         f"cov={dr['test_coverage']:.2f}")
            print(line)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=SOURCES, default="auto",
                        help="real local data, tracked surrogate, or auto (real if present)")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    source = resolve_source(args.data)
    print(f"Data source: {source}")
    models = fit_all(load_pairs(source))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models.json", "eval_report.json"):
        with open(args.out_dir / name, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)
    write_rules(models, args.out_dir / "conversion_rules.md")
    print(f"\nWrote {args.out_dir / 'models.json'}")


if __name__ == "__main__":
    main()
