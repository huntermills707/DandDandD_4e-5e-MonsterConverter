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
from sklearn.linear_model import LinearRegression, RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.datasets import SOURCES, load_pairs, resolve_source  # noqa: E402

MODELS_DIR = ROOT / "models"

ALPHA = 0.20  # nominal miscoverage; 80% scale-adaptive prediction intervals
N_FOLDS = 5
N_REPEATS = 10
MIN_N = 15

# Each spec: direction, target field on the target side, and source features:
# the target's counterpart stat(s) in the other edition plus the level/CR
# scale (4e builds half its level into attacks and defenses). Simpler specs
# with more features fit no better under grouped CV (models/experiments.py).
SCALE = {"4e_to_5e": "level", "5e_to_4e": "cr_num"}
ABILITIES = ("str", "dex", "con", "int", "wis", "cha")
COUNTERPART = {
    "4e_to_5e": {
        "cr_num": [], "hp": ["hp"], "ac": ["ac"], "attack_bonus": ["attack_bonus"],
        "damage_avg": ["damage_avg"],
        **{f"ability_{a}": [f"ability_{a}"] for a in ABILITIES},
        # 5e lists only proficient saves: modifier + proficiency
        **{f"save_{a}": [f"ability_{a}"] for a in ("dex", "con", "wis", "cha")},
    },
    "5e_to_4e": {
        "level": [], "hp": ["hp"], "ac": ["ac"],
        # 4e defenses use the better of two abilities
        "fort": ["ability_str", "ability_con"], "ref": ["ability_dex"],
        "will": ["ability_wis", "ability_cha"],
        "attack_bonus": ["attack_bonus"], "damage_avg": ["damage_avg"],
        **{f"ability_{a}": [f"ability_{a}"] for a in ABILITIES},
    },
}
MODEL_SPECS = [
    {"direction": d, "target": t, "features": feats + [SCALE[d]]}
    for d, targets in COUNTERPART.items() for t, feats in targets.items()
]


ROLE_TIERS = ("minion", "elite", "solo")  # 4e role tiers; "standard" is the reference


def role_tier(role):
    r = (role or "").lower()
    return next((t for t in ROLE_TIERS if t in r), "standard")


class LinearDesign:
    """A linear model in (optionally) transformed space.

        z    = intercept + sum(coef_k * t_k(x_k))     t_k = log1p or identity
        pred = inverse_target(z)                       log -> expm1(z)

    fit:      "nonneg" (coefficients and intercept >= 0), "ols", or "ridge"
              (standardized features, penalty by efficient leave-one-out;
              coefficients reported back in original units)
    interval: "relative" [pred(1-q), pred(1+q)], "absolute" [pred-q, pred+q],
              "log" [expm1(z-q), expm1(z+q)] (multiplicative), or "scaled"
              pred +- q(|pred| + c) with c = median |target| in training:
              absolute-like for small values, relative-like for large ones
    """

    def __init__(self, name="nonneg_linear", fit="nonneg", interval="relative",
                 log_target=False, log_features=(), clip_min=0.0, round_sig=None):
        self.name, self.fit_kind, self.interval = name, fit, interval
        self.round_sig = round_sig
        self.log_target, self.log_features, self.clip_min = log_target, set(log_features), clip_min
        self.features = None

    def __call__(self):  # acts as its own factory: make_model() -> fresh copy
        d = LinearDesign(self.name, self.fit_kind, self.interval, self.log_target,
                         self.log_features, self.clip_min, self.round_sig)
        d.features = self.features
        return d

    def _tx(self, X):
        X = np.array(X, dtype=float)
        for k, f in enumerate(self.features):
            if f in self.log_features:
                X[:, k] = np.log1p(np.maximum(X[:, k], 0))
        return X

    def fit(self, X, y):
        self.scale_offset_ = float(np.median(np.abs(y)))
        if self.round_sig:
            self.scale_offset_ = float(f"{self.scale_offset_:.2g}")
        Xt = self._tx(X)
        yt = np.log1p(np.maximum(y, 0)) if self.log_target else y
        if Xt.shape[1] == 0:  # intercept-only rule: the typical value
            self.intercept_, self.coef_ = float(np.mean(yt)), np.zeros(0)
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
            m = LinearRegression().fit(Xt, yt)
            self.intercept_, self.coef_ = float(m.intercept_), m.coef_
        if self.round_sig:
            # hand-usable equation: coefficients to `round_sig` significant
            # figures, then the intercept refit for them and rounded to 0.1
            self.coef_ = np.array([float(f"{c:.{self.round_sig}g}") for c in self.coef_])
            self.intercept_ = round(float(np.mean(yt - Xt @ self.coef_)), 1)
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
                "round_sig": self.round_sig,
                "log_target": self.log_target,
                "log_features": sorted(self.log_features & set(self.features)),
                "clip_min": self.clip_min,
                "interval_offset": round(getattr(self, "scale_offset_", 0.0), 4)}


NonNegLinear = LinearDesign()  # the original design

# Chosen design (models/experiments.py, D9), picked for being easy to apply
# by hand at no measurable cost in accuracy: counterpart stat + level/CR +
# 4e role adjustments, least squares, coefficients rounded to 2 significant
# figures, interval  result +- q * (result + c).
ROLE_FEATURES = ["role_minion", "role_elite", "role_solo"]
DEFAULT_DESIGN = LinearDesign("explainable", fit="ols", interval="scaled", round_sig=2)


def source_target(spec):
    if spec["direction"] == "4e_to_5e":
        return "4e", "5e"
    return "5e", "4e"


def feature_value(pair, source_prefix, feature):
    """Source-side stat, or a 4e role indicator ("role_elite", ...). Role
    comes from the 4e side in both directions: for 5e -> 4e it is the role
    the converted monster should have."""
    if feature.startswith("role_"):
        return float(role_tier(pair["4e"].get("role")) == feature[5:])
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


def oof_predictions(X, y, groups, make_model, n_folds, seed):
    pred = np.empty_like(y)
    for test in grouped_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = make_model().fit(X[train], y[train]).predict(X[test])
    return pred


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
        pred = np.empty_like(y)
        base = np.empty_like(y)
        lo = np.empty_like(y)
        hi = np.empty_like(y)
        for f, test in enumerate(grouped_folds(groups, n_folds, seed)):
            train = np.setdiff1d(np.arange(len(y)), test)
            model = make_model().fit(X[train], y[train])
            pred[test] = model.predict(X[test])
            base[test] = y[train].mean()
            # nested: interval width chosen on the training folds only
            inner = oof_predictions(X[train], y[train], groups[train], make_model,
                                    n_folds, seed * 31 + f)
            q = conformal_q(model.residual(y[train], inner), alpha)
            lo[test], hi[test] = model.bounds(pred[test], q)
        oof_preds.append(pred)

        m = is_same
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
    return {k: np.array(v) for k, v in per.items()}, np.array(oof_preds)


def cv_mae(X, y, groups, is_same, make_model, n_repeats=3, n_folds=N_FOLDS, seed=500):
    """Mean held-out MAE on same-creature pairs (light CV for selection)."""
    out = []
    for rep in range(n_repeats):
        pred = oof_predictions(X, y, groups, make_model, n_folds, seed + rep)
        out.append(np.abs(y - pred)[is_same].mean())
    return float(np.mean(out))


def select_features(pairs, spec, make_model, counterparts, tol=0.01):
    """Backward elimination toward short, sensible rules.

    1. A counterpart stat with a negative coefficient is dropped (a stat
       shouldn't convert in the opposite direction; it means the scale
       term is doing the work).
    2. Then, repeatedly, the term whose removal raises CV error least is
       dropped while that rise stays under `tol` (1%).
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

    while True:
        X, y, g, same, model = data(features)
        model.fit(X, y)
        neg = [f for f, c in zip(features, model.coef_) if f in counterparts and c < 0]
        if not neg:
            break
        features.remove(neg[0])
        dropped.append(neg[0])

    X, y, g, same, model = data(features)
    current = cv_mae(X, y, g, same, model)
    while features:
        trials = []
        for f in features:
            rest = [x for x in features if x != f]
            Xr, yr, gr, sr, mr = data(rest)
            trials.append((cv_mae(Xr, yr, gr, sr, mr), f))
        best_mae, best_f = min(trials)
        if best_mae > current * (1 + tol):
            break
        features.remove(best_f)
        dropped.append(best_f)
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
    final = make_model().fit(X, y)
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
    "damage_avg": "damage", "fort": "Fort", "ref": "Ref", "will": "Will",
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
    """e.g. 'HP = 0.19·HP + 1.3·level - 12 (minion -4, elite -9.7, solo +59)'."""
    terms, roles = [], []
    for c, f in zip(model["coef"], model["features"]):
        if f.startswith("role_"):
            roles.append(f"{f[5:]} {'+' if c >= 0 else '-'}{num(abs(c))}")
            continue
        label = short_label(f)
        if f in set(model.get("log_features", [])):
            label = f"ln(1+{label})"
        coef = "" if abs(abs(c) - 1) < 0.005 else f"{num(abs(c))}·"
        terms.append(f"{'-' if c < 0 else '+'} {coef}{label}")
    inter = model["intercept"]
    if inter or not terms:
        terms.append(f"{'-' if inter < 0 else '+'} {abs(inter):g}")
    rhs = " ".join(terms)
    rhs = rhs[2:] if rhs.startswith("+ ") else "-" + rhs[2:]
    eq = f"{short_label(model['target'])} = {rhs}"
    if roles:
        eq += f"  ({', '.join(roles)})"
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
    ]
    for direction, title in (("4e_to_5e", "4e → 5e"), ("5e_to_4e", "5e → 4e")):
        lines += [f"## {title}", "", "| Stat | Rule | 80% range | Error | R² |",
                  "|---|---|---|---|---|"]
        for m in models.values():
            if m["direction"] != direction:
                continue
            rule = m["equation"].split(" = ", 1)[1]
            lines.append(f"| {short_label(m['target'])} | {rule} | {interval_string(m)} | "
                         f"{m['mae'] / m['baseline_mae']:.0%} | {m['r2']:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_all(pairs, make_model=DEFAULT_DESIGN, extra_features=ROLE_FEATURES, select=True,
            verbose=True):
    models = {}
    for spec in MODEL_SPECS:
        key = f"{spec['direction']}/{spec['target']}"
        counterparts = COUNTERPART[spec["direction"]][spec["target"]]
        spec = dict(spec, features=spec["features"] + list(extra_features))
        dropped = []
        if select:
            kept, dropped = select_features(pairs, spec, make_model, counterparts)
            spec = dict(spec, features=kept)
        result = fit_spec(pairs, spec, make_model)
        if result is not None:
            result["dropped_features"] = dropped
        if result is None:
            if verbose:
                print(f"  skipping {key}: not enough data")
            continue
        result["equation"] = equation_string(result)
        result["interval_rule"] = interval_string(result)
        models[key] = result
        if verbose:
            print(f"{key:26s} R²={result['r2']:6.2f}±{result['r2_sd']:.2f}  "
                  f"MAE={result['mae']:7.2f} (base {result['baseline_mae']:7.2f})  "
                  f"cov={result['test_coverage']:.2f}")
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
