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
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.datasets import SOURCES, load_pairs, resolve_source  # noqa: E402

MODELS_DIR = ROOT / "models"

ALPHA = 0.20  # nominal miscoverage; 80% scale-adaptive prediction intervals
N_FOLDS = 5
N_REPEATS = 10
MIN_N = 15

# Each spec: direction, target field on target side, list of source features.
MODEL_SPECS = [
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


class NonNegLinear:
    """Linear model with non-negative coefficients and intercept, so
    predictions from non-negative stats are never negative."""

    name = "nonneg_linear"

    def fit(self, X, y):
        self.model = LinearRegression(fit_intercept=False, positive=True)
        self.model.fit(np.column_stack([np.ones(len(X)), X]), y)
        self.intercept_ = float(self.model.coef_[0])
        self.coef_ = self.model.coef_[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


def source_target(spec):
    if spec["direction"] == "4e_to_5e":
        return "4e", "5e"
    return "5e", "4e"


def extract(pairs, source_prefix, target_prefix, features, target):
    """Rows with the target and every feature present.
    Returns X, y, groups, is_same (same-creature pair)."""
    X, y, groups, same = [], [], [], []
    for k, p in enumerate(pairs):
        src, tgt = p[source_prefix], p[target_prefix]
        if tgt.get(target) is None or any(src.get(f) is None for f in features):
            continue
        X.append([src[f] for f in features])
        y.append(tgt[target])
        groups.append(p.get("group", k))
        same.append(p.get("match_type", "same") == "same")
    return (np.array(X, dtype=float).reshape(-1, len(features)), np.array(y, dtype=float),
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


def relative(y, pred):
    return np.abs(y - pred) / np.maximum(np.abs(pred), 1e-6)


def oof_predictions(X, y, groups, make_model, n_folds, seed):
    pred = np.empty_like(y)
    for test in grouped_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred[test] = make_model().fit(X[train], y[train]).predict(X[test])
    return pred


def cross_validate(X, y, groups, is_same, make_model, alpha=ALPHA,
                   n_folds=N_FOLDS, n_repeats=N_REPEATS):
    """Repeated grouped CV. Returns per-repeat metric arrays and the pooled
    out-of-fold relative residuals (all pairs) for the final interval."""
    per = {k: [] for k in ("r2", "mae", "baseline_mae", "coverage", "rel_width")}
    oof_rel = []
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
            q = conformal_q(relative(y[train], inner), alpha)
            lo[test] = np.maximum(pred[test] * (1 - q), 0.0)
            hi[test] = pred[test] * (1 + q)
        oof_rel.append(relative(y, pred))

        m = is_same
        yt, pt = y[m], pred[m]
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        per["r2"].append(1 - ((yt - pt) ** 2).sum() / ss_tot if ss_tot else np.nan)
        per["mae"].append(np.abs(yt - pt).mean())
        per["baseline_mae"].append(np.abs(yt - base[m]).mean())
        per["coverage"].append(((yt >= lo[m]) & (yt <= hi[m])).mean())
        per["rel_width"].append(np.mean((hi[m] - lo[m]) / np.maximum(np.abs(yt), 1e-6)))
    return {k: np.array(v) for k, v in per.items()}, np.concatenate(oof_rel)


def fit_spec(pairs, spec, make_model=NonNegLinear, alpha=ALPHA,
             n_folds=N_FOLDS, n_repeats=N_REPEATS):
    source_prefix, target_prefix = source_target(spec)
    features, target = spec["features"], spec["target"]
    X, y, groups, is_same = extract(pairs, source_prefix, target_prefix, features, target)
    if len(y) < MIN_N or is_same.sum() < N_FOLDS or len(np.unique(groups)) < N_FOLDS:
        return None

    per, oof_rel = cross_validate(X, y, groups, is_same, make_model, alpha, n_folds, n_repeats)
    final = make_model().fit(X, y)

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
        "estimator": make_model.name,
        "n": int(len(y)),
        "n_eval": int(is_same.sum()),
        "n_groups": int(len(np.unique(groups))),
        "cv": f"grouped {n_folds}-fold x{n_repeats}, scored on same-creature pairs",
        "intercept": round(float(final.intercept_), 4),
        "coef": [round(float(c), 6) for c in final.coef_],
        "alpha": alpha,
        "conformal_quantile_relative": round(conformal_q(oof_rel, alpha), 4),
        "r2": r2, "r2_sd": r2_sd,
        "mae": mae, "mae_sd": mae_sd,
        "baseline_mae": base, "baseline_mae_sd": base_sd,
        "test_coverage": cov, "test_coverage_sd": cov_sd,
        "interval_rel_width": round(float(np.mean(per["rel_width"])), 3),
    }


def equation_string(model):
    parts = []
    for c, f in zip(model["coef"], model["features"]):
        label = f.replace("ability_", "").replace("save_", "save ").replace("cr_num", "CR").replace("damage_avg", "dmg").replace("attack_bonus", "atk")
        parts.append(f"{c:.4f}·{label}")
    eq = " + ".join(parts)
    inter = model["intercept"]
    sign = "+" if inter >= 0 else "-"
    target = model["target"].replace("ability_", "").replace("cr_num", "CR").replace("damage_avg", "dmg").replace("attack_bonus", "atk")
    return f"{target} = {eq} {sign} {abs(inter):.4f}"


def fit_all(pairs, make_model=NonNegLinear, verbose=True):
    models = {}
    for spec in MODEL_SPECS:
        key = f"{spec['direction']}/{spec['target']}"
        result = fit_spec(pairs, spec, make_model)
        if result is None:
            if verbose:
                print(f"  skipping {key}: not enough data")
            continue
        result["equation"] = equation_string(result)
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
    print(f"\nWrote {args.out_dir / 'models.json'}")


if __name__ == "__main__":
    main()
