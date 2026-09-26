"""Writing and checking ``submission.csv``.

Format from the dataset README: ``;`` as the separator, UTF-8, a header, exactly two columns
``sample_id;prediction``, every ``sample_id`` from ``validate/points.csv`` once, prediction
in seconds (positive means late).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(sample_ids, predictions, path: Path | str) -> Path:
    """Write predictions in the platform format. Refuses NaN and infinite values."""
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain NaN or inf")
    frame = pd.DataFrame({"sample_id": list(sample_ids), "prediction": np.round(predictions, 1)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=";", index=False, encoding="utf-8")
    return path


def check_submission(path: Path | str, expected_ids) -> list[str]:
    """Problems with a submission file, an empty list if it is fine."""
    try:
        frame = pd.read_csv(path, sep=";", dtype={"sample_id": str})
    except (OSError, pd.errors.ParserError) as exc:
        return [f"cannot read {path}: {exc}"]

    if list(frame.columns) != ["sample_id", "prediction"]:
        return [f"columns must be sample_id;prediction, got {';'.join(frame.columns)}"]

    problems = []
    expected = set(map(str, expected_ids))
    got = frame["sample_id"]
    if got.duplicated().any():
        problems.append(f"{got.duplicated().sum()} duplicated sample_id")
    missing = expected - set(got)
    if missing:
        problems.append(f"{len(missing)} sample_id missing, e.g. {sorted(missing)[0]}")
    extra = set(got) - expected
    if extra:
        problems.append(f"{len(extra)} unknown sample_id, e.g. {sorted(extra)[0]}")
    values = pd.to_numeric(frame["prediction"], errors="coerce")
    if not np.isfinite(values).all():
        problems.append("prediction has empty, non-numeric or infinite values")
    return problems
