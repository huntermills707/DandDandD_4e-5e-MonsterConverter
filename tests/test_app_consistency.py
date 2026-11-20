"""The app's prediction/interval math must match the fitted models."""

import sys

import numpy as np
import pytest

from models.fit_models import DEFAULT_DESIGN, MODEL_SPECS, extract, source_target
from pipeline.datasets import ROOT, load_pairs

sys.path.insert(0, str(ROOT / "app"))
import app  # noqa: E402


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda s: f"{s['direction']}/{s['target']}")
def test_app_matches_fitter(spec):
    key = f"{spec['direction']}/{spec['target']}"
    model = app.MODELS[key]
    features = model["features"]  # after pruning

    src, tgt = source_target(spec)
    X, y, *_ = extract(load_pairs("surrogate"), src, tgt, features, spec["target"])
    fitted = DEFAULT_DESIGN()
    fitted.features = features
    fitted.fit(X, y)
    pred = fitted.predict(X)
    lo, hi = fitted.bounds(pred, model["conformal_quantile"])

    for row, p, low, high in zip(X, pred, lo, hi):
        got = app.predict_with_interval(model, dict(zip(features, row)))
        assert np.allclose(got, (p, low, high), atol=1e-2, rtol=1e-4)
