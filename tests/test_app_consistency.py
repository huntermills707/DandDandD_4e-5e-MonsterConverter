"""The app's prediction/interval math must match the fitted models."""

import sys

import numpy as np
import pytest

from models.fit_models import (DEFAULT_DESIGN, MODEL_SPECS, extract, feature_value, role_flags,
                               role_tier, rule_pairs, source_target)
from pipeline.datasets import ROOT, load_pairs

sys.path.insert(0, str(ROOT / "app"))
import app  # noqa: E402


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: f"{s['direction']}/{s['target']}")
def test_app_matches_fitter(spec):
    key = f"{spec['direction']}/{spec['target']}"
    model = app.MODELS[key]
    features = model["features"]  # after pruning

    src, tgt = source_target(spec)
    pairs = rule_pairs(load_pairs("surrogate"), spec)
    X, y, *_ = extract(pairs, src, tgt, features, spec["target"])
    fitted = DEFAULT_DESIGN()
    fitted.features = features
    fitted.fit(X, y)
    pred = fitted.predict(X)
    lo, hi = fitted.bounds(pred, model["conformal_quantile"])

    usable = [p for p in pairs if p[tgt].get(spec["target"]) is not None
              and all(feature_value(p, src, f) is not None for f in features)]
    for p, row, pr, low, high in zip(usable, X, pred, lo, hi):
        # the app builds interaction terms from typed stats + the role selector
        state = {**p[src], **{f"role_{t}": float(role_tier(p["4e"].get("role")) == t)
                              for t in ("minion", "elite", "solo")}}
        values = app.feature_values(features, state)
        assert np.allclose([values[f] for f in features], row)
        got = app.predict_with_interval(model, values, role_tier(p["4e"].get("role")))
        assert np.allclose(got, (pr, low, high), atol=1e-2, rtol=1e-4)


def test_minion_hp_is_fixed():
    model = app.MODELS["5e_to_4e/hp"]
    assert model["fixed_by_role"] == {"minion": 1.0}
    assert app.predict_with_interval(model, None, "minion") == (1.0, 1.0, 1.0)
    assert "minion = 1" in model["equation"]


def test_role_flags_and_interactions():
    assert role_flags("Elite Controller (Leader)") == {"elite", "controller", "leader"}
    assert role_flags("Solo Soldier") == {"solo", "soldier"}
    pair = {"4e": {"role": "Solo Lurker", "level": 7}, "5e": {"cr_num": 4.0}}
    assert feature_value(pair, "5e", "role_solo*cr_num") == 4.0
    assert feature_value(pair, "5e", "role_elite*cr_num") == 0.0
    assert feature_value(pair, "5e", "role_lurker") == 1.0
    assert feature_value(pair, "5e", "role_solo*hp") is None  # missing stat
    assert app.feature_values(["role_solo*cr_num", "cr_num"],
                              {"role_solo": 1.0, "cr_num": "4"}) == {"role_solo*cr_num": 4.0,
                                                                     "cr_num": 4.0}
    assert app.feature_values(["role_solo*cr_num"], {"role_solo": 1.0, "cr_num": None}) is None
