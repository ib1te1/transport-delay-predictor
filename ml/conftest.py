# pytest puts the directory of this file on sys.path, so "busdelay" imports without
# installing the package, and tests/ on it too for the "world" helper.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))
