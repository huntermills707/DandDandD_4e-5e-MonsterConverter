"""The app's prediction/interval math must match the fitted models."""

import sys

import numpy as np
import pytest

from models.fit_models import (DEFAULT_DESIGN, MODEL_SPECS, LinearDesign, extract,
                               feature_value, role_flags, role_tier, rule_pairs, source_target,
                               weighted_median)
from pipeline.datasets import ROOT, load_pairs

sys.path.insert(0, str(ROOT / "app"))
import app  # noqa: E402


def roles(pair):
    """The app's role-selector inputs for a pair's 4e role."""
    role = pair["4e"].get("role")
    combat = next((r for r in app.COMBAT_ROLES if r in role_flags(role)), "skirmisher")
    return app.role_state(role_tier(role), combat)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: f"{s['direction']}/{s['target']}")
def test_app_matches_fitter(spec):
    key = f"{spec['direction']}/{spec['target']}"
    model = app.MODELS[key]
    features = model["features"]  # after pruning

    src, tgt = source_target(spec)
    pairs = rule_pairs(load_pairs("surrogate"), spec)
    X, y, _, same = extract(pairs, src, tgt, features, spec["target"])
    fitted = DEFAULT_DESIGN()
    fitted.features = features
    fitted.fit(X, y, same)
    pred = fitted.predict(X)
    lo, hi = fitted.bounds(pred, model["conformal_quantile"])

    usable = [p for p in pairs if p[tgt].get(spec["target"]) is not None
              and all(feature_value(p, src, f) is not None for f in features)]
    for p, row, pr, low, high in zip(usable, X, pred, lo, hi):
        # the app builds interaction terms from typed stats + the role selector
        # and the target level/CR the GM types in
        scale = app.TARGET_SCALE[spec["direction"]]
        state = {**p[src], scale: p[tgt].get(scale), **roles(p)}
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
    # the target edition's level/CR is read from the other side
    assert feature_value(pair, "5e", "level") == 7
    assert feature_value(pair, "5e", "role_solo*level") == 7.0
    assert feature_value(pair, "4e", "cr_num") == 4.0
    assert feature_value(pair, "5e", "role_elite*cr_num") == 0.0
    assert feature_value(pair, "5e", "role_lurker") == 1.0
    assert feature_value(pair, "5e", "role_solo*hp") is None  # missing stat
    assert app.feature_values(["role_solo*cr_num", "cr_num"],
                              {"role_solo": 1.0, "cr_num": "4"}) == {"role_solo*cr_num": 4.0,
                                                                     "cr_num": 4.0}
    assert app.feature_values(["role_solo*cr_num"], {"role_solo": 1.0, "cr_num": None}) is None


def test_max_feature():
    """max(a,b): the better of two abilities, in the fit and in the app."""
    pair = {"4e": {"ability_str": 18, "ability_dex": 21, "ability_con": None}, "5e": {}}
    assert feature_value(pair, "4e", "max(ability_str,ability_dex)") == 21
    assert feature_value(pair, "4e", "max(ability_str,ability_con)") is None  # missing stat
    assert app.input_stats("max(ability_str,ability_dex)") == ["ability_str", "ability_dex"]
    assert app.feature_values(["max(ability_str,ability_dex)"],
                              {"ability_str": "18", "ability_dex": "21"}) == {
        "max(ability_str,ability_dex)": 21.0}
    assert app.feature_values(["max(ability_str,ability_dex)"],
                              {"ability_str": "18", "ability_dex": None}) is None
    assert app.feature_label("max(ability_str,ability_dex)") == "max(STR, DEX)"


def test_related_weight_and_weighted_median():
    assert weighted_median([1, 2, 3, 100], [1, 1, 1, 1]) == 2
    assert weighted_median([1, 2, 3, 100], [1, 1, 1, 10]) == 100
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 30.0])
    same = np.array([True, True, True, False])
    for fit in ("ols", "huber"):
        m = LinearDesign(fit=fit, related_weight=0.0)
        m.features = ["x"]
        m.fit(X, y, same)  # the related outlier is left out: exact line
        assert np.allclose([m.intercept_, *m.coef_], [0.0, 1.0], atol=1e-3)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: f"{s['direction']}/{s['target']}")
def test_app_direct_rules_match_fitter(spec):
    """With the target level/CR left blank, the app applies each stat's
    direct rule (source level/CR), reproducing its fit."""
    model = app.MODELS[f"{spec['direction']}/{spec['target']}"]
    if "direct" not in model:
        pytest.skip("the level/CR rule itself")
    direct = model["direct"]
    src, tgt = source_target(spec)
    pairs = rule_pairs(load_pairs("surrogate"), spec)
    X, y, _, same = extract(pairs, src, tgt, direct["features"], spec["target"])
    fitted = DEFAULT_DESIGN()
    fitted.features = direct["features"]
    fitted.fit(X, y, same)
    pred = fitted.predict(X)
    lo, hi = fitted.bounds(pred, direct["conformal_quantile"])
    usable = [p for p in pairs if p[tgt].get(spec["target"]) is not None
              and all(feature_value(p, src, f) is not None for f in direct["features"])]
    scale = app.TARGET_SCALE[spec["direction"]]
    for p, pr, low, high in zip(usable, pred, lo, hi):
        state = {**p[src], scale: "", **roles(p)}
        assert not app.target_scale_given(spec["direction"], state)
        rule = app.rule_for(model, False)
        values = app.feature_values(rule["features"], state)
        got = app.predict_with_interval(rule, values, role_tier(p["4e"].get("role")))
        assert np.allclose(got, (pr, low, high), atol=1e-2, rtol=1e-4)


def test_target_scale_switches_rules():
    model = app.MODELS["5e_to_4e/ac"]
    assert app.target_scale_given("5e_to_4e", {"level": 7})
    assert not app.target_scale_given("5e_to_4e", {"level": ""})
    assert app.rule_for(model, True) is model
    assert app.rule_for(model, False) is model["direct"]
    assert "level" in model["features"] and "level" not in model["direct"]["features"]


def test_combat_role_state():
    st = app.role_state("elite", "brute")
    assert st["role_elite"] == 1.0 and st["role_brute"] == 1.0
    assert sum(st.values()) == 2.0
    assert sum(app.role_state("standard", "skirmisher").values()) == 0.0
