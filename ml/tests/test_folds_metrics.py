import numpy as np
import pandas as pd
import pytest

from busdelay.folds import block_ids, group_folds
from busdelay.metrics import mae, score_estimate, summary


def test_blocks_break_on_part_vehicle_and_gaps():
    points = pd.DataFrame(
        {
            "tr_id": [1, 1, 1, 1, 1, 2, 2],
            "T": [0, 300, 600, 900, 3600, 0, 300],
            "part": ["train", "train", "test", "test", "test", "train", "train"],
        }
    )
    blocks = block_ids(points)
    assert blocks.tolist() == [1, 1, 2, 2, 3, 4, 4]


def test_blocks_follow_the_index_of_unsorted_points():
    points = pd.DataFrame({"tr_id": [1, 1], "T": [300, 0], "part": ["test", "train"]})
    assert block_ids(points).tolist() == [2, 1]


def test_group_folds_keep_groups_whole_and_balanced():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), rng.integers(1, 20, 40))
    folds = group_folds(groups, n_folds=5, seed=3)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    sizes = np.bincount(folds)
    assert len(sizes) == 5 and sizes.max() - sizes.min() <= 20
    assert (group_folds(groups, n_folds=5, seed=3) == folds).all()


def test_group_folds_needs_enough_groups():
    with pytest.raises(ValueError):
        group_folds([1, 1, 2], n_folds=5)


def test_mae_and_score():
    y = np.array([100.0, -50.0, 20.0, 0.0])
    cur = np.array([60.0, 0.0, 40.0, 10.0])
    assert mae(y, y) == 0
    assert mae(y, 0) == 42.5
    assert score_estimate(y, np.zeros(4)) == pytest.approx(0.0)
    assert score_estimate(y, y) == pytest.approx(1 / (1 - 0.76))
    # MAE_TARGET = 0.76 * 42.5 = 32.3 should score exactly 1
    assert score_estimate(y, y + np.array([32.3, -32.3, 32.3, -32.3])) == pytest.approx(1.0)
    s = summary(y, cur, cur)
    assert s["rows"] == 4 and s["mae"] == mae(y, cur) and s["vs_persistence"] == 0
