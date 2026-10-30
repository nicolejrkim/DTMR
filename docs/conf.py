"""Sphinx configuration for the DTMR documentation (Furo theme, Markdown via MyST).

Build locally:
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html
"""
from __future__ import annotations

import re
from pathlib import Path

project = "DTMR"
author = "Jaeryeong Kim, Taerim Yoon, Jin Cheng, Sungjoon Choi, Stelian Coros"
copyright = "2026, the DTMR authors"

extensions = ["myst_parser", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "README.md"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "DTMR"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/nicolejrkim/DTMR",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Pages are Markdown that also reads well on GitHub, so their links point at repo
# files (`../src/dtmr/examples/x.yaml`, `HUMAN_MOTION.md`). A link to a page of this site
# stays a page link; a link to anything else in the repo becomes a GitHub URL.
REPO_URL = "https://github.com/nicolejrkim/DTMR/blob/main/"
DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
LINK = re.compile(r'(\]\(|src=")(?!https?://|#|mailto:)([^)\s"]+)')


def _rewrite(text: str, page_dir: Path) -> str:
    """Rewrite the links of a page that lives in `page_dir`."""
    def fix(m):
        target, _, anchor = m.group(2).partition("#")
        anchor = "#" + anchor if anchor else ""
        path = (page_dir / target).resolve()
        if not path.exists() or ROOT not in path.parents:
            return m.group(0)
        if DOCS in path.parents and "_build" not in path.parts:
            return m.group(0)                   # a doc page or an image under docs/
        return f"{m.group(1)}{REPO_URL}{path.relative_to(ROOT).as_posix()}{anchor}"
    return LINK.sub(fix, text)


def _on_source(app, docname, source):
    source[0] = _rewrite(source[0], (DOCS / docname).parent)


def _copy_assets(app, exception):
    """Media referenced from raw HTML (galleries, videos) is not collected by Sphinx."""
    if exception is None and app.builder.format == "html":
        import shutil
        shutil.copytree(DOCS / "assets", Path(app.outdir) / "assets", dirs_exist_ok=True)


def setup(app):
    app.connect("source-read", _on_source)
    app.connect("build-finished", _copy_assets)
