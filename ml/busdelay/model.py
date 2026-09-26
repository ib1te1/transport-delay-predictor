"""CatBoost models on top of the feature table.

Two models use the same features:

* the regressor predicts the delay in seconds with MAE loss, this is what the platform
  scores. By default it learns the residual over the two-line baseline
  (:class:`busdelay.baselines.SplitLinear`) and the prediction is baseline + model;
* the classifier gives the probability that the bus is more than 2 minutes late at the
  target stop (``late`` in the organisers' ``target_class``). The dashboard shows risk with it.

The number of trees is picked by cross-validation: every fold is trained with all the trees
and scored after each one on its held-out real vehicles, the curves are summed over the
folds and the minimum is taken. The final models are then trained on all labeled points
with that many trees.
"""

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from . import __version__
from .baselines import SplitLinear
from .features import FEATURES
from .folds import block_ids, group_folds
from .metrics import classifier_summary, summary

# "late" in the organisers' target_class
LATE_S = 120.0
VEHICLE_COLUMN = "vehicle"
# quantiles of the out-of-fold residuals used as the forecast interval
INTERVAL_QUANTILES = (0.1, 0.9)

REGRESSOR_FILE = "regressor.cbm"
CLASSIFIER_FILE = "late.cbm"
META_FILE = "meta.json"
OOF_FILE = "oof.parquet"


@dataclass
class TrainConfig:
    """Settings of one training run, all can be set from the command line."""

    iterations: int = 3000
    depth: int = 6
    learning_rate: float = 0.03
    l2_leaf_reg: float = 3.0
    residual: bool = True
    vehicle: bool = False
    synthetic_weight: float = 1.0
    classifier: bool = True
    folds: int = 5
    # "runs" - like the organisers' split; "vehicle" - whole vehicles held out;
    # "route" - a real vehicle together with its synthetic copies, the honest check
    group: str = "runs"
    seed: int = 0
    threads: int = -1
    gpu: bool = False

    def columns(self) -> list[str]:
        return FEATURES + ([VEHICLE_COLUMN] if self.vehicle else [])

    def catboost_params(self, iterations: int | None = None) -> dict:
        return {
            "iterations": iterations or self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_seed": self.seed,
            "thread_count": self.threads,
            "verbose": False,
            "allow_writing_files": False,
            # on GPU CatBoost scores MAE on the CPU every 5 trees by default, the curves
            # that pick the number of trees need every tree
            **({"task_type": "GPU", "devices": "0", "metric_period": 1} if self.gpu else {}),
        }


def make_matrix(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Model input in the given column order. ``vehicle`` is ``tr_id`` as a category."""
    numeric = [c for c in columns if c != VEHICLE_COLUMN]
    X = table[numeric].astype(float)
    if VEHICLE_COLUMN in columns:
        X[VEHICLE_COLUMN] = table["tr_id"].astype(str).to_numpy()
    return X[columns]


def _pool(table, columns, label=None, weight=None) -> Pool:
    categorical = [c for c in columns if c == VEHICLE_COLUMN]
    return Pool(make_matrix(table, columns), label=label, weight=weight, cat_features=categorical)


def _weights(table: pd.DataFrame, config: TrainConfig) -> np.ndarray:
    return np.where(table["synthetic"].to_numpy(dtype=bool), config.synthetic_weight, 1.0)


@dataclass
class DelayModel:
    """Trained models plus what is needed to use them."""

    regressor: CatBoostRegressor
    classifier: CatBoostClassifier | None
    columns: list[str]
    lines: dict | None = None
    interval: tuple[float, float] = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def baseline(self, table: pd.DataFrame) -> np.ndarray:
        if self.lines is None:
            return np.zeros(len(table))
        base = SplitLinear()
        base.lines = self.lines
        return base.predict(table)

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """``delay_s`` with its interval and ``late_prob`` (if there is a classifier)."""
        X = make_matrix(table, self.columns)
        delay = self.baseline(table) + self.regressor.predict(X)
        out = pd.DataFrame(
            {
                "delay_s": delay,
                "delay_lo_s": delay + self.interval[0],
                "delay_hi_s": delay + self.interval[1],
            },
            index=table.index,
        )
        if self.classifier is not None:
            out["late_prob"] = self.classifier.predict_proba(X)[:, 1]
        return out

    def importance(self, top: int = 20) -> list[tuple[str, float]]:
        values = self.regressor.get_feature_importance()
        order = np.argsort(-values)[:top]
        return [(self.columns[i], round(float(values[i]), 2)) for i in order]

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.regressor.save_model(str(directory / REGRESSOR_FILE))
        if self.classifier is not None:
            self.classifier.save_model(str(directory / CLASSIFIER_FILE))
        meta = {
            **self.meta,
            "columns": self.columns,
            "lines": None
            if self.lines is None
            else {"regular": self.lines[False], "after_layover": self.lines[True]},
            "interval": list(self.interval),
            "has_classifier": self.classifier is not None,
        }
        (directory / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "DelayModel":
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        regressor = CatBoostRegressor()
        regressor.load_model(str(directory / REGRESSOR_FILE))
        classifier = None
        if meta.get("has_classifier"):
            classifier = CatBoostClassifier()
            classifier.load_model(str(directory / CLASSIFIER_FILE))
        lines = meta.get("lines")
        if lines is not None:
            lines = {False: tuple(lines["regular"]), True: tuple(lines["after_layover"])}
        return cls(
            regressor=regressor,
            classifier=classifier,
            columns=meta["columns"],
            lines=lines,
            interval=tuple(meta["interval"]),
            meta=meta,
        )


def fit_models(
    table: pd.DataFrame, config: TrainConfig, iterations: int, iterations_late: int | None
) -> DelayModel:
    """Train the regressor (and the classifier) on all rows of ``table``."""
    y = table["target"].to_numpy(dtype=float)
    weight = _weights(table, config)
    columns = config.columns()

    lines = None
    offset = np.zeros(len(table))
    if config.residual:
        base = SplitLinear().fit(table, y)
        lines = base.lines
        offset = base.predict(table)

    regressor = CatBoostRegressor(
        loss_function="MAE", eval_metric="MAE", **config.catboost_params(iterations)
    )
    regressor.fit(_pool(table, columns, y - offset, weight))

    classifier = None
    late = (y > LATE_S).astype(int)
    if config.classifier and iterations_late and 0 < late.sum() < len(late):
        classifier = CatBoostClassifier(
            loss_function="Logloss", **config.catboost_params(iterations_late)
        )
        classifier.fit(_pool(table, columns, late, weight))
    return DelayModel(regressor, classifier, columns, lines)


@dataclass
class CVResult:
    folds: np.ndarray
    base: np.ndarray
    oof: np.ndarray
    oof_late: np.ndarray | None
    iterations: int
    iterations_late: int | None


def cross_validate(table: pd.DataFrame, config: TrainConfig) -> CVResult:
    """Out-of-fold predictions and the best number of trees, see the module docstring."""
    y = table["target"].to_numpy(dtype=float)
    late = (y > LATE_S).astype(int)
    real = ~table["synthetic"].to_numpy(dtype=bool)
    weight = _weights(table, config)
    columns = config.columns()
    if config.group not in ("runs", "vehicle", "route"):
        raise ValueError(f"group must be runs, vehicle or route, got {config.group!r}")
    if config.group == "runs":
        folds = group_folds(block_ids(table), n_folds=config.folds, seed=config.seed)
    elif config.group == "route":
        folds = group_folds(table["route"], n_folds=config.folds, seed=config.seed)
    else:
        # real and synthetic vehicles are spread separately, so every fold gets real ones
        vehicles = table["tr_id"].to_numpy()
        folds = np.zeros(len(table), dtype=int)
        for rows in (real, ~real):
            if len(np.unique(vehicles[rows])) >= config.folds:
                folds[rows] = group_folds(vehicles[rows], n_folds=config.folds, seed=config.seed)
    # nothing to classify if no one (or everyone) is late
    with_classifier = config.classifier and 0 < late.sum() < len(late)

    base = np.zeros(len(table))
    curves, late_curves, regressors, classifiers = [], [], [], []
    for k in range(config.folds):
        train = folds != k
        held = (folds == k) & real
        if not held.any():
            raise ValueError(f"fold {k} has no real vehicles to score on")

        lines = SplitLinear().fit(table[train], y[train])
        base[folds == k] = lines.predict(table[folds == k])
        offset_train = lines.predict(table[train]) if config.residual else 0.0
        offset_held = base[held] if config.residual else 0.0

        regressor = CatBoostRegressor(
            loss_function="MAE", eval_metric="MAE", **config.catboost_params()
        )
        regressor.fit(
            _pool(table[train], columns, y[train] - offset_train, weight[train]),
            eval_set=_pool(table[held], columns, y[held] - offset_held),
            use_best_model=False,
        )
        curves.append(np.asarray(regressor.get_evals_result()["validation"]["MAE"]) * held.sum())
        regressors.append(regressor)

        if with_classifier:
            classifier = CatBoostClassifier(loss_function="Logloss", **config.catboost_params())
            classifier.fit(
                _pool(table[train], columns, late[train], weight[train]),
                eval_set=_pool(table[held], columns, late[held]),
                use_best_model=False,
            )
            curve = classifier.get_evals_result()["validation"]["Logloss"]
            late_curves.append(np.asarray(curve) * held.sum())
            classifiers.append(classifier)

    iterations = int(np.argmin(np.sum(curves, axis=0))) + 1
    iterations_late = int(np.argmin(np.sum(late_curves, axis=0))) + 1 if late_curves else None

    oof = np.zeros(len(table))
    oof_late = np.zeros(len(table)) if classifiers else None
    for k in range(config.folds):
        rows = folds == k
        X = make_matrix(table[rows], columns)
        offset = base[rows] if config.residual else 0.0
        oof[rows] = offset + regressors[k].predict(X, ntree_end=iterations)
        if classifiers:
            oof_late[rows] = classifiers[k].predict_proba(X, ntree_end=iterations_late)[:, 1]
    return CVResult(folds, base, oof, oof_late, iterations, iterations_late)


def train(
    table: pd.DataFrame,
    config: TrainConfig,
    stream_test: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> tuple[DelayModel, dict, pd.DataFrame]:
    """Cross-validate, check on the organisers' split, then fit on everything.

    The control model is fit on the train rows of ``table`` and scored on its test rows, or
    on ``test`` when the table has none (training on train only). ``stream_test`` is the
    test part with ``cur_dev_s`` estimated from GPS: the accuracy to expect on the stream.

    Returns the final model, the metrics and the out-of-fold predictions.
    """
    started = datetime.now(UTC)
    cv = cross_validate(table, config)

    y = table["target"].to_numpy(dtype=float)
    cur = table["cur_dev_s"].fillna(0.0).to_numpy()
    real = ~table["synthetic"].to_numpy(dtype=bool)
    layover = table["new_trip_ahead"].to_numpy() > 0
    metrics = {
        "cv": {
            "persistence": summary(y[real], cur[real], cur[real]),
            "split_linear": summary(y[real], cv.base[real], cur[real]),
            "model": summary(y[real], cv.oof[real], cur[real]),
            "model / layover": summary(
                y[real & layover], cv.oof[real & layover], cur[real & layover]
            ),
            "model / no layover": summary(
                y[real & ~layover], cv.oof[real & ~layover], cur[real & ~layover]
            ),
        },
        "iterations": cv.iterations,
        "iterations_late": cv.iterations_late,
    }
    if cv.oof_late is not None:
        metrics["late"] = classifier_summary(y[real] > LATE_S, cv.oof_late[real])

    part = table["part"].to_numpy()
    control = fit_models(table[part == "train"], config, cv.iterations, cv.iterations_late)
    if (part == "test").any():
        test = table[part == "test"]
    elif test is None:
        raise ValueError("no test rows to score the control model on")
    y_test = test["target"].to_numpy(dtype=float)
    cur_test = test["cur_dev_s"].fillna(0.0).to_numpy()
    metrics["control"] = {
        "persistence": summary(y_test, cur_test, cur_test),
        "model": summary(y_test, control.predict(test)["delay_s"], cur_test),
    }
    if stream_test is not None:
        y_stream = stream_test["target"].to_numpy(dtype=float)
        gps_cur = stream_test["cur_dev_s"].fillna(0.0).to_numpy()
        metrics["control"]["persistence, cur_dev from GPS"] = summary(y_stream, gps_cur, gps_cur)
        metrics["control"]["model, cur_dev from GPS"] = summary(
            y_stream, control.predict(stream_test)["delay_s"], gps_cur
        )

    residuals = y[real] - cv.oof[real]
    model = fit_models(table, config, cv.iterations, cv.iterations_late)
    model.interval = tuple(float(q) for q in np.quantile(residuals, INTERVAL_QUANTILES))
    model.meta = {
        "created": started.isoformat(timespec="seconds"),
        "config": asdict(config),
        "metrics": metrics,
        "versions": {
            "busdelay": __version__,
            "python": platform.python_version(),
            "catboost": catboost.__version__,
        },
    }
    model.meta["importance"] = model.importance()

    oof = table[["sample_id", "tr_id", "part", "target", "synthetic", "cur_dev_s"]].copy()
    oof["fold"] = cv.folds
    oof["base"] = cv.base
    oof["oof"] = cv.oof
    if cv.oof_late is not None:
        oof["oof_late"] = cv.oof_late
    return model, metrics, oof
