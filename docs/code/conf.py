"""Sphinx settings for the code reference."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = os.environ.get("CODE_DOC_TARGET", "home")
PATHS = {
    "api": ROOT / "services" / "api",
    "ingest": ROOT / "services" / "ingest",
    "matcher": ROOT / "services" / "matcher",
    "predictor": ROOT / "services" / "predictor",
    "contracts": ROOT / "packages" / "contracts" / "src",
    "common": ROOT / "packages" / "common" / "src",
    "ml": ROOT / "ml",
}

if TARGET in PATHS:
    sys.path.insert(0, str(PATHS[TARGET]))
sys.path.insert(0, str(ROOT / "packages" / "contracts" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(ROOT / "ml"))

if TARGET in {"api", "ingest", "matcher", "predictor"}:
    # Load FastAPI before autodoc inspects the shared Pydantic models.
    import fastapi  # noqa: F401

project = "Предиктор задержек транспорта"
language = "ru"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "alabaster"
autodoc_typehints = "description"
exclude_patterns = []
