"""Simple predictors the model has to beat.

They take the feature frame from :func:`busdelay.features.build_features` and have the same
``fit(frame, y)`` / ``predict(frame)`` methods as the model wrapper.
"""

import numpy as np
import pandas as pd


def fit_l1_line(x, y, slopes=None) -> tuple[float, float]:
    """``a, b`` minimising ``mean(|y - (a * x + b)|)``.

    The slope is searched on a grid. For a fixed slope the best intercept is the median of
    ``y - a * x``, so only one number is actually searched.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if slopes is None:
        slopes = np.linspace(-0.5, 1.5, 201)
    best = (np.inf, 0.0, 0.0)
    for a in slopes:
        b = float(np.median(y - a * x))
        error = float(np.abs(y - a * x - b).mean())
        if error < best[0]:
            best = (error, float(a), b)
    return best[1], best[2]


def _cur_dev(frame: pd.DataFrame) -> np.ndarray:
    return frame["cur_dev_s"].fillna(0.0).to_numpy(dtype=float)


class Zero:
    """No delay at all. Scores exactly 0 on the platform."""

    name = "zero"

    def fit(self, frame, y):
        return self

    def predict(self, frame):
        return np.zeros(len(frame))


class Persistence:
    """The delay stays what ``cur_dev_s`` says. This is ``sample_submission.csv``."""

    name = "persistence"

    def fit(self, frame, y):
        return self

    def predict(self, frame):
        return _cur_dev(frame)


class Linear:
    """``a * cur_dev_s + b``: persistence shrunk towards the typical delay."""

    name = "linear"

    def fit(self, frame, y):
        self.a, self.b = fit_l1_line(_cur_dev(frame), y)
        return self

    def predict(self, frame):
        return self.a * _cur_dev(frame) + self.b


class SplitLinear:
    """Two lines: for targets in the next trip (after a layover) and for the rest.

    At a terminal the bus usually makes up its delay, so ``cur_dev_s`` means much less for
    a target that lies behind a layover.
    """

    name = "split_linear"

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        after_layover = frame["new_trip_ahead"].to_numpy() > 0
        self.lines = {}
        for flag in (False, True):
            rows = after_layover == flag
            self.lines[flag] = (
                fit_l1_line(_cur_dev(frame)[rows], y[rows]) if rows.any() else (1.0, 0.0)
            )
        return self

    def predict(self, frame):
        after_layover = frame["new_trip_ahead"].to_numpy() > 0
        x = _cur_dev(frame)
        result = np.empty(len(frame))
        for flag, (a, b) in self.lines.items():
            rows = after_layover == flag
            result[rows] = a * x[rows] + b
        return result


BASELINES = {cls.name: cls for cls in (Zero, Persistence, Linear, SplitLinear)}
