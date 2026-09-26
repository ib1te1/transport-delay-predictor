import numpy as np
import pandas as pd
import pytest

from busdelay.baselines import Linear, Persistence, SplitLinear, Zero, fit_l1_line
from busdelay.submission import check_submission, write_submission


def test_l1_line_recovers_the_line():
    x = np.linspace(-100, 300, 50)
    a, b = fit_l1_line(x, 0.55 * x + 8)
    assert a == pytest.approx(0.55, abs=0.011)
    assert b == pytest.approx(8, abs=2)


def test_l1_line_ignores_outliers():
    x = np.linspace(0, 100, 101)
    y = 0.5 * x
    y[::10] += 5000
    a, b = fit_l1_line(x, y)
    assert a == pytest.approx(0.5, abs=0.02)


def test_simple_baselines():
    frame = pd.DataFrame({"cur_dev_s": [10.0, np.nan, -30.0], "new_trip_ahead": [0.0, 0.0, 1.0]})
    assert Zero().fit(frame, None).predict(frame).tolist() == [0, 0, 0]
    assert Persistence().fit(frame, None).predict(frame).tolist() == [10, 0, -30]


def test_split_linear_uses_two_lines():
    x = np.tile(np.linspace(0, 300, 30), 2)
    layover = np.repeat([0.0, 1.0], 30)
    y = np.where(layover > 0, 20.0, x)
    frame = pd.DataFrame({"cur_dev_s": x, "new_trip_ahead": layover})
    split = SplitLinear().fit(frame, y)
    assert np.abs(split.predict(frame) - y).max() < 5
    single = Linear().fit(frame, y)
    assert np.abs(single.predict(frame) - y).mean() > np.abs(split.predict(frame) - y).mean()


def test_submission_round_trip(tmp_path):
    path = write_submission(["a_1", "b_2"], [120.04, -30.0], tmp_path / "sub.csv")
    assert path.read_text(encoding="utf-8").splitlines() == [
        "sample_id;prediction",
        "a_1;120.0",
        "b_2;-30.0",
    ]
    assert check_submission(path, ["a_1", "b_2"]) == []


def test_submission_problems_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("sample_id;prediction\na;1\na;2\nz;x\n", encoding="utf-8")
    problems = " ".join(check_submission(path, ["a", "b"]))
    assert "duplicated" in problems
    assert "missing" in problems
    assert "unknown" in problems
    assert "non-numeric" in problems

    comma = tmp_path / "comma.csv"
    comma.write_text("sample_id,prediction\na,1\n", encoding="utf-8")
    assert check_submission(comma, ["a"])


def test_nan_predictions_are_not_written(tmp_path):
    with pytest.raises(ValueError):
        write_submission(["a"], [np.nan], tmp_path / "x.csv")
