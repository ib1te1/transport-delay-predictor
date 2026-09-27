"""Build the code reference with each service in an isolated Sphinx process."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "code"
OUTPUT = ROOT / "data" / "docs" / "code"
TARGETS = {
    "api": (ROOT / "services" / "api" / "app", "app"),
    "ingest": (ROOT / "services" / "ingest" / "app", "app"),
    "matcher": (ROOT / "services" / "matcher" / "app", "app"),
    "predictor": (ROOT / "services" / "predictor" / "app", "app"),
    "contracts": (ROOT / "packages" / "contracts" / "src" / "contracts", "contracts"),
    "common": (ROOT / "packages" / "common" / "src" / "common", "common"),
    "ml": (ROOT / "ml" / "busdelay", "busdelay"),
}


def write_target_pages(source: Path, name: str, package_dir: Path, package: str) -> None:
    modules = [package]
    modules.extend(
        f"{package}.{path.stem}"
        for path in sorted(package_dir.glob("*.py"))
        if path.stem not in {"__init__", "__main__"} and not path.stem.startswith("_")
    )
    pages = []
    for module in modules:
        page = module.replace(".", "-")
        pages.append(page)
        title = module
        (source / f"{page}.rst").write_text(
            f"{title}\n{'=' * len(title)}\n\n"
            f".. automodule:: {module}\n   :members:\n   :undoc-members:\n"
            "   :show-inheritance:\n",
            encoding="utf-8",
        )
    title = f"{name}: справочник Python API"
    toctree = "\n".join(f"   {page}" for page in pages)
    (source / "index.rst").write_text(
        f"{title}\n{'=' * len(title)}\n\n"
        "Публичные классы и функции пакета. Названия модулей соответствуют исходному коду.\n\n"
        f".. toctree::\n   :maxdepth: 2\n\n{toctree}\n",
        encoding="utf-8",
    )


def build(source: Path, output: Path, target: str) -> None:
    env = os.environ.copy()
    env["CODE_DOC_TARGET"] = target
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            "-c",
            str(SOURCE),
            str(source),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build(SOURCE, OUTPUT, "home")
    for name, (package_dir, package) in TARGETS.items():
        with tempfile.TemporaryDirectory(prefix=f"sphinx-{name}-") as directory:
            source = Path(directory)
            write_target_pages(source, name, package_dir, package)
            build(source, OUTPUT / name, name)
    print(f"Documentation: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
