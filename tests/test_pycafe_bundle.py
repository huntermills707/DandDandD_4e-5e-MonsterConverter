"""The flat PyCafe bundles load their models and styles without the repo."""

import base64
import gzip
import importlib.util
import json
import sys
from urllib.parse import unquote

import pytest

from deploy import build_pycafe


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("pycafe")
    links = build_pycafe.build(out)
    return out, links


def load_app(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Dash looks the module up by name
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    return module


@pytest.mark.parametrize("name,models_attr", [("converter", "MODELS"),
                                              ("scaler", "SCALE_MODELS")])
def test_bundle_loads_models_and_inlines_css(bundle, name, models_attr):
    out, _ = bundle
    folder = out / name
    assert sorted(p.name for p in folder.iterdir()) == sorted(build_pycafe.bundle_files(name))
    module = load_app(folder / "app.py", f"pycafe_{name}")
    assert module.HERE == folder.resolve()
    assert getattr(module, models_attr), "models not found beside the bundled app"
    assert ".role-note" in module.app.index_string  # styles.css inlined


def test_snippet_link_round_trips(bundle):
    _, links = bundle
    for name, link in links.items():
        prefix = build_pycafe.SNIPPET_URL + "#c="
        assert link.startswith(prefix)
        payload = json.loads(gzip.decompress(base64.b64decode(unquote(link[len(prefix):]))))
        files = build_pycafe.bundle_files(name)
        assert payload["code"] == files["app.py"]
        assert payload["requirements"] == files["requirements.txt"]
        assert {f["name"]: f["content"] for f in payload["files"]} == {
            n: t for n, t in files.items() if n not in ("app.py", "requirements.txt")}


def test_repo_app_uses_assets_folder():
    """Run from the repo, Dash serves app/assets/ and nothing is inlined."""
    sys.path.insert(0, str(build_pycafe.ROOT / "app"))
    import app
    assert app.MODELS_PATH == build_pycafe.ROOT / "models" / "models.json"
    assert "<style>" not in app.app.index_string


def test_minion_note_only_for_4e_to_5e():
    sys.path.insert(0, str(build_pycafe.ROOT / "app"))
    import app
    text, style = app.update_role_note("4e_to_5e", "minion")
    assert text == app.MINION_NOTE and style == {}
    for direction, role in [("4e_to_5e", "standard"), ("5e_to_4e", "minion")]:
        assert app.update_role_note(direction, role) == ("", {"display": "none"})
