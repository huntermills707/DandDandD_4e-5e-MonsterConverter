#!/usr/bin/env python3
"""Fit per-field linear regressions for both 4e -> 5e and 5e -> 4e.

Each model uses a small set of source-edition covariates.  Conformal
prediction gives per-prediction intervals with a nominal coverage level
(default 90%).  Results are saved to models/models.json for the Dash app.
"""

import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
TRAINING_PATH = ROOT / "data" / "processed" / "training_pairs.jsonl"
MODELS_PATH = ROOT / "models" / "models.json"
REPORT_PATH = ROOT / "models" / "eval_report.json"

ALPHA = 0.20  # nominal miscoverage; 80% scale-adaptive prediction intervals

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


def load_pairs():
    with open(TRAINING_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def source_target(spec):
    if spec["direction"] == "4e_to_5e":
        return "4e", "5e"
    return "5e", "4e"


def extract(pairs, source_prefix, target_prefix, features, target):
    X, y = [], []
    for p in pairs:
        src = p[source_prefix]
        tgt = p[target_prefix]
        if tgt.get(target) is None:
            continue
        if any(src.get(f) is None for f in features):
            continue
        X.append([src[f] for f in features])
        y.append(tgt[target])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def split_three(X, y, train_frac=0.6, cal_frac=0.2, random_state=42):
    n = len(y)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    train_idx = idx[:n_train]
    cal_idx = idx[n_train : n_train + n_cal]
    test_idx = idx[n_train + n_cal :]
    return (
        X[train_idx], y[train_idx],
        X[cal_idx], y[cal_idx],
        X[test_idx], y[test_idx],
    )


def conformal_quantile(residuals, alpha):
    """Conservative split-conformal quantile."""
    n = len(residuals)
    if n == 0:
        return 0.0
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(residuals, q_level, method="higher"))


def conformal_relative_quantile(y_true, y_pred, alpha):
    """Conservative split-conformal quantile of relative residuals.

    Returns a fraction q such that the interval
        [pred * (1 - q), pred * (1 + q)]
    has approximate (1 - alpha) coverage on the calibration set.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    rel = np.abs(y_true - y_pred) / np.maximum(np.abs(y_pred), 1e-6)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(rel, q_level, method="higher"))


def fit_spec(pairs, spec):
    source_prefix, target_prefix = source_target(spec)
    features = spec["features"]
    target = spec["target"]
    X, y = extract(pairs, source_prefix, target_prefix, features, target)
    n = len(y)
    if n < 15:
        return None

    # Augment with a constant column so we can constrain the intercept to be
    # non-negative as well. All source features are non-negative, so this
    # guarantees predictions are never negative.
    X_aug = np.column_stack([np.ones(n, dtype=float), X])
    X_tr, y_tr, X_cal, y_cal, X_te, y_te = split_three(X_aug, y)
    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X_tr, y_tr)

    y_cal_pred = model.predict(X_cal)
    q_rel = conformal_relative_quantile(y_cal, y_cal_pred, ALPHA)

    y_te_pred = model.predict(X_te)
    coverage = np.mean(
        (y_te >= y_te_pred * (1 - q_rel)) & (y_te <= y_te_pred * (1 + q_rel))
    )
    baseline_pred = np.full_like(y_te, y_tr.mean())

    return {
        "direction": spec["direction"],
        "source_edition": source_prefix,
        "target_edition": target_prefix,
        "target": target,
        "features": features,
        "n": n,
        "n_train": len(y_tr),
        "n_cal": len(y_cal),
        "n_test": len(y_te),
        "intercept": round(float(model.coef_[0]), 4),
        "coef": [round(float(c), 6) for c in model.coef_[1:]],
        "alpha": ALPHA,
        "conformal_quantile_relative": round(q_rel, 4),
        "test_coverage": round(float(coverage), 3),
        "r2": round(r2_score(y_te, y_te_pred), 3),
        "mae": round(mean_absolute_error(y_te, y_te_pred), 3),
        "baseline_mae": round(mean_absolute_error(y_te, baseline_pred), 3),
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


def main():
    pairs = load_pairs()
    models = {}
    for spec in MODEL_SPECS:
        key = f"{spec['direction']}/{spec['target']}"
        result = fit_spec(pairs, spec)
        if result is None:
            print(f"  skipping {key}: not enough data")
            continue
        result["equation"] = equation_string(result)
        models[key] = result
        print(f"{key}: R²={result['r2']}, MAE={result['mae']}, "
              f"coverage={result['test_coverage']}, q_rel={result['conformal_quantile_relative']}")

    MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_PATH, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)
    print(f"\nWrote {MODELS_PATH}")


if __name__ == "__main__":
    main()
