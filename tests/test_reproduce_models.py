"""Fitting on the tracked surrogate reproduces the committed models exactly.

This is what makes the published models auditable without the rule books:
anyone can rerun the fits on data/surrogate and get the same equations.
"""

import json

from models import fit_models, fit_scale_models
from pipeline.datasets import ROOT


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_conversion_models_reproduce(tmp_path):
    fit_models.main(["--data", "surrogate", "--out-dir", str(tmp_path)])
    assert load(tmp_path / "models.json") == load(ROOT / "models" / "models.json")


def test_scale_models_reproduce(tmp_path):
    fit_scale_models.main(["--data", "surrogate", "--out-dir", str(tmp_path)])
    assert load(tmp_path / "scale_models.json") == load(ROOT / "models" / "scale_models.json")
