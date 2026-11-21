#!/usr/bin/env python3
"""Bundle the Dash apps for PyCafe (https://py.cafe), which runs Python in
the browser (Pyodide): no server to host.

Each app becomes a flat folder, ready to upload to a PyCafe Dash project:

    dist/pycafe/converter/  app.py  models.json        styles.css  requirements.txt
    dist/pycafe/scaler/     app.py  scale_models.json  styles.css  requirements.txt

The apps find their model JSON and styles.css beside themselves when run
from such a folder (see data_file / inline_css in app/app.py). The script
also prints a PyCafe snippet link per app with every file embedded
(dist/pycafe/links.md): open it, check the app, and save it as a project.

    .venv/bin/python deploy/build_pycafe.py
"""

import argparse
import base64
import gzip
import json
import shutil
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
SNIPPET_URL = "https://py.cafe/snippet/dash/v1"

# Both apps import only json/math/pathlib/sys and Dash. Unpinned, so PyCafe
# picks the Dash its Pyodide build supports.
REQUIREMENTS = "dash\n"

APPS = {
    "converter": {"source": ROOT / "app" / "app.py", "data": ROOT / "models" / "models.json"},
    "scaler": {"source": ROOT / "app" / "scale_app.py",
               "data": ROOT / "models" / "scale_models.json"},
}
STYLES = ROOT / "app" / "assets" / "styles.css"


def bundle_files(name):
    """{file name: text} for one app's flat bundle; the entry point is app.py."""
    spec = APPS[name]
    return {
        "app.py": spec["source"].read_text(encoding="utf-8"),
        spec["data"].name: spec["data"].read_text(encoding="utf-8"),
        "styles.css": STYLES.read_text(encoding="utf-8"),
        "requirements.txt": REQUIREMENTS,
    }


def snippet_link(files):
    """PyCafe snippet link with every file embedded: gzip-compressed JSON
    {code, requirements, files: [{name, content}]}, base64, URL-quoted
    (https://py.cafe/docs/api)."""
    payload = {
        "code": files["app.py"],
        "requirements": files["requirements.txt"],
        "files": [{"name": n, "content": text} for n, text in files.items()
                  if n not in ("app.py", "requirements.txt")],
    }
    compressed = gzip.compress(json.dumps(payload).encode("utf-8"))
    return f"{SNIPPET_URL}#c={quote(base64.b64encode(compressed).decode('ascii'))}"


def build(out_dir):
    out_dir = Path(out_dir)
    links = {}
    for name in APPS:
        files = bundle_files(name)
        target = out_dir / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for file_name, text in files.items():
            (target / file_name).write_text(text, encoding="utf-8")
        links[name] = snippet_link(files)
    lines = ["# PyCafe snippet links", "",
             "Open a link, check the app runs, then save it as a PyCafe project.", ""]
    for name, link in links.items():
        lines += [f"## {name}", "", link, ""]
    (out_dir / "links.md").write_text("\n".join(lines), encoding="utf-8")
    return links


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default=ROOT / "dist" / "pycafe", type=Path)
    args = parser.parse_args(argv)
    links = build(args.out_dir)
    for name in APPS:
        print(f"{name}: {args.out_dir / name}/  (snippet link: {len(links[name]):,} chars)")
    print(f"links: {args.out_dir / 'links.md'}")


if __name__ == "__main__":
    main()
