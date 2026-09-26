"""MAE and the platform score.

The platform computes::

    score = (mae_zero - MAE) / (mae_zero - MAE_TARGET), clipped to [0, 1]

where ``mae_zero`` is the MAE of predicting zero. ``MAE_TARGET`` is not published. The
dataset README says the ``cur_dev_s`` baseline scores about 0.40 on validate; on the test
part (same vehicles, same day) that happens with ``MAE_TARGET = 0.76 * mae_zero``, so
:func:`score_estimate` uses this ratio. It is a rough guide, the model is chosen by MAE.
"""

import numpy as np
import pandas as pd

TARGET_TO_ZERO = 0.76


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def score_estimate(y_true, y_pred) -> float:
    """Estimated platform score, see the module docstring. Not clipped."""
    zero = mae(y_true, 0.0)
    if zero == 0:
        return float("nan")
    return (zero - mae(y_true, y_pred)) / (zero * (1 - TARGET_TO_ZERO))


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve through ranks (ties get the average rank)."""
    labels = np.asarray(labels, dtype=bool)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank().to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def classifier_summary(labels, probability) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "auc": roc_auc(labels, p),
        "logloss": float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()),
        "brier": float(((p - labels) ** 2).mean()),
        "positive_rate": float(labels.mean()),
        "rows": int(len(labels)),
    }


def summary(y_true, y_pred, cur_dev) -> dict[str, float]:
    """The numbers printed for every model and baseline.

    ``vs_persistence`` is the relative MAE change against predicting ``cur_dev_s``,
    negative is better.
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    persistence = mae(y_true, cur_dev)
    return {
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "p90": float(np.percentile(np.abs(error), 90)),
        "vs_persistence": float(np.abs(error).mean() / persistence - 1) if persistence else np.nan,
        "score": score_estimate(y_true, y_pred),
        "rows": int(len(error)),
    }
