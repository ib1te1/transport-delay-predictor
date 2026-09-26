"""Cross-validation folds that look like the organisers' split.

Train, test and validate points come from the same day and the same buses. The organisers
cut the day of every bus into runs of consecutive 5-minute points and gave each run to one
part. Folds keep such runs whole: neighbouring points differ by five minutes and share most
of their features, so splitting a run would make the score look better than it is.
"""

import numpy as np
import pandas as pd

# A run ends where the part changes or where the points stop for longer than this.
MAX_GAP_S = 15 * 60


def block_ids(points: pd.DataFrame) -> pd.Series:
    """Number runs of points: same vehicle, same part, no gap longer than ``MAX_GAP_S``.

    ``points`` needs ``tr_id``, ``T`` and ``part``. The result is aligned with its index.
    """
    ordered = points.sort_values(["tr_id", "T"], kind="stable")
    new_block = (
        ordered["tr_id"].ne(ordered["tr_id"].shift())
        | ordered["part"].ne(ordered["part"].shift())
        | ordered["T"].diff().gt(MAX_GAP_S)
    )
    return new_block.cumsum().reindex(points.index)


def group_folds(groups, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign a fold to every row so that a group never ends up in two folds.

    Groups are shuffled and then handed out largest first, each to the fold that currently
    has the fewest rows, so folds come out about the same size.
    """
    groups = np.asarray(groups)
    names, sizes = np.unique(groups, return_counts=True)
    if len(names) < n_folds:
        raise ValueError(f"{len(names)} groups is not enough for {n_folds} folds")

    order = np.random.default_rng(seed).permutation(len(names))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_rows = np.zeros(n_folds, dtype=int)
    fold_of = {}
    for i in order:
        fold = int(np.argmin(fold_rows))
        fold_of[names[i]] = fold
        fold_rows[fold] += sizes[i]
    return np.array([fold_of[g] for g in groups])
