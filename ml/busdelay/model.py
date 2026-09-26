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
with that many trees, with several seeds averaged into one model.

The table may hold every point twice, with the organisers' ``cur_dev_s`` and with the one
estimated from GPS (column ``hint``, ``given`` or ``gps``). Metrics are then reported for
both, the live stream only has the GPS one. Rows with ``hint`` ``outage`` are copies with
the last minutes of telemetry cut (:mod:`busdelay.outage`). They only add training data:
they get their own metrics, but the number of trees is picked without them.
"""

import json
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, sum_models

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

HINTS = ("given", "gps", "mix")
GPS_SUFFIX = ", cur_dev from GPS"
OUTAGE_SUFFIX = ", telemetry cut 1-15 min"
SUFFIXES = {"given": "", "gps": GPS_SUFFIX, "outage": OUTAGE_SUFFIX}

REGRESSOR_FILE = "regressor.cbm"
CLASSIFIER_FILE = "late.cbm"
META_FILE = "meta.json"


@dataclass
class TrainConfig:
    """Settings of one training run, all can be set from the command line."""

    iterations: int = 3000
    depth: int = 4
    learning_rate: float = 0.03
    l2_leaf_reg: float = 30.0
    residual: bool = True
    vehicle: bool = False
    synthetic_weight: float = 1.0
    # weight of the outage copies (hint "outage"), 0 when the table has none
    outage_weight: float = 0.0
    classifier: bool = True
    folds: int = 5
    # "runs" - like the organisers' split; "vehicle" - whole vehicles held out;
    # "route" - a real vehicle together with its synthetic copies, the honest check
    group: str = "runs"
    seed: int = 0
    # the final models are the average of this many seeds, starting from ``seed``
    seeds: int = 1
    # which cur_dev_s the table has: "given", "gps" or "mix" (both), only recorded here
    hint: str = "given"
    threads: int = -1
    gpu: bool = False

    def columns(self) -> list[str]:
        """Feature columns the models get."""
        return FEATURES + ([VEHICLE_COLUMN] if self.vehicle else [])

    def catboost_params(self, iterations: int | None = None, seed: int | None = None) -> dict:
        """CatBoost settings for one model of this run."""
        return {
            "iterations": iterations or self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_seed": self.seed if seed is None else seed,
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


def hint_rows(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row masks by where ``cur_dev_s`` came from. A table without ``hint`` is all given."""
    if "hint" not in table:
        return {"given": np.ones(len(table), dtype=bool)}
    hint = table["hint"].to_numpy()
    return {h: hint == h for h in SUFFIXES if (hint == h).any()}


def _outage(table: pd.DataFrame) -> np.ndarray:
    if "hint" not in table:
        return np.zeros(len(table), dtype=bool)
    return table["hint"].to_numpy() == "outage"


def _weights(table: pd.DataFrame, config: TrainConfig) -> np.ndarray:
    weight = np.where(table["synthetic"].to_numpy(dtype=bool), config.synthetic_weight, 1.0)
    return np.where(_outage(table), config.outage_weight, weight)


def fit_lines(table: pd.DataFrame, y: np.ndarray) -> SplitLinear:
    """The two lines, fit without the outage copies: their ``cur_dev_s`` is often minutes
    off and would flatten the lines for everyone."""
    keep = ~_outage(table)
    return SplitLinear().fit(table[keep], np.asarray(y)[keep])


@dataclass
class DelayModel:
    """Trained models plus what is needed to use them.

    The delay is ``baseline_weight * two lines + trees``: 1 for a residual model, less for a
    blend of residual and plain models (:func:`blend`).
    """

    regressor: CatBoostRegressor
    classifier: CatBoostClassifier | None
    columns: list[str]
    lines: dict | None = None
    interval: tuple[float, float] = (0.0, 0.0)
    baseline_weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def baseline(self, table: pd.DataFrame) -> np.ndarray:
        """The two-line part of the prediction, already weighted."""
        if self.lines is None:
            return np.zeros(len(table))
        base = SplitLinear()
        base.lines = self.lines
        return self.baseline_weight * base.predict(table)

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
        """The ``top`` features by CatBoost importance of the regressor."""
        values = self.regressor.get_feature_importance()
        order = np.argsort(-values)[:top]
        return [(self.columns[i], round(float(values[i]), 2)) for i in order]

    def save(self, directory: Path | str) -> Path:
        """Write the models and ``meta.json`` into ``directory``."""
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
            "baseline_weight": self.baseline_weight,
            "has_classifier": self.classifier is not None,
        }
        (directory / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "DelayModel":
        """Read a model written by :meth:`save`."""
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
            baseline_weight=float(meta.get("baseline_weight", 1.0)),
            meta=meta,
        )


def _fit_seeds(cls, pool: Pool, config: TrainConfig, iterations: int, **params):
    """``config.seeds`` models with consecutive seeds, averaged into one of class ``cls``."""
    models = [
        cls(**params, **config.catboost_params(iterations, seed=config.seed + i)).fit(pool)
        for i in range(config.seeds)
    ]
    if len(models) == 1:
        return models[0]
    return _average(cls, models, [1 / len(models)] * len(models))


def _average(cls, models: list, weights: list[float]):
    """Weighted sum of CatBoost models as one model of class ``cls``.

    For classifiers the raw values (log-odds) are summed.
    """
    merged = sum_models(models, weights=weights)
    # the sum has no training params and CatBoost warns about it on every load; the first
    # model's params stand in, what was summed is in meta.json
    merged.get_metadata()["params"] = models[0].get_metadata()["params"]
    # sum_models gives a plain CatBoost, a file round trip makes it a regressor or classifier
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "merged.cbm")
        merged.save_model(path)
        model = cls()
        model.load_model(path)
    return model


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
        base = fit_lines(table, y)
        lines = base.lines
        offset = base.predict(table)

    regressor = _fit_seeds(
        CatBoostRegressor,
        _pool(table, columns, y - offset, weight),
        config,
        iterations,
        loss_function="MAE",
        eval_metric="MAE",
    )

    classifier = None
    late = (y > LATE_S).astype(int)
    if config.classifier and iterations_late and 0 < late.sum() < len(late):
        classifier = _fit_seeds(
            CatBoostClassifier,
            _pool(table, columns, late, weight),
            config,
            iterations_late,
            loss_function="Logloss",
        )
    return DelayModel(regressor, classifier, columns, lines)


@dataclass
class CVResult:
    """Out-of-fold results: fold of every row, the two lines, the model, the late probability
    and the chosen numbers of trees."""

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
        # the curves that pick the number of trees; outage copies would pull it their way
        held = (folds == k) & real & ~_outage(table)
        if not held.any():
            raise ValueError(f"fold {k} has no real vehicles to score on")

        lines = fit_lines(table[train], y[train])
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


OOF_COLUMNS = ["sample_id", "tr_id", "part", "target", "synthetic", "cur_dev_s"]


def cv_scores(oof: pd.DataFrame) -> dict:
    """Metrics of the out-of-fold predictions on real vehicles, per ``cur_dev_s`` source."""
    y = oof["target"].to_numpy(dtype=float)
    cur = oof["cur_dev_s"].fillna(0.0).to_numpy()
    real = ~oof["synthetic"].to_numpy(dtype=bool)
    layover = oof["layover"].to_numpy(dtype=bool)
    pred = oof["oof"].to_numpy(dtype=float)
    base = oof["base"].to_numpy(dtype=float)
    scores = {}
    for hint, rows in hint_rows(oof).items():
        suffix = SUFFIXES[hint]
        r = real & rows
        scores.update(
            {
                f"persistence{suffix}": summary(y[r], cur[r], cur[r]),
                f"split_linear{suffix}": summary(y[r], base[r], cur[r]),
                f"model{suffix}": summary(y[r], pred[r], cur[r]),
                f"model / layover{suffix}": summary(
                    y[r & layover], pred[r & layover], cur[r & layover]
                ),
                f"model / no layover{suffix}": summary(
                    y[r & ~layover], pred[r & ~layover], cur[r & ~layover]
                ),
            }
        )
    return scores


def control_scores(oof: pd.DataFrame) -> dict:
    """Metrics of the control model (fit on train) on the test rows, per ``cur_dev_s`` source."""
    y = oof["target"].to_numpy(dtype=float)
    cur = oof["cur_dev_s"].fillna(0.0).to_numpy()
    pred = oof["control"].to_numpy(dtype=float)
    scored = ~np.isnan(pred)
    scores = {}
    for hint, rows in hint_rows(oof).items():
        r = scored & rows
        if r.any():
            suffix = SUFFIXES[hint]
            scores[f"persistence{suffix}"] = summary(y[r], cur[r], cur[r])
            scores[f"model{suffix}"] = summary(y[r], pred[r], cur[r])
    return scores


def _interval(oof: pd.DataFrame, served: np.ndarray) -> tuple[float, float]:
    residuals = oof["target"].to_numpy(dtype=float)[served] - oof["oof"].to_numpy()[served]
    return tuple(float(q) for q in np.quantile(residuals, INTERVAL_QUANTILES))


def _meta(model: DelayModel, started: datetime, metrics: dict, **extra) -> dict:
    return {
        "created": started.isoformat(timespec="seconds"),
        **extra,
        "metrics": metrics,
        "versions": {
            "busdelay": __version__,
            "python": platform.python_version(),
            "catboost": catboost.__version__,
        },
        "importance": model.importance(),
    }


def train(
    table: pd.DataFrame,
    config: TrainConfig,
    stream_test: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> tuple[DelayModel, dict, pd.DataFrame]:
    """Cross-validate, check on the organisers' split, then fit on everything.

    The control model is fit on the train rows of ``table`` and scored on ``test`` (the test
    part with the organisers' ``cur_dev_s``), or on the given-hint test rows of ``table`` if
    ``test`` is not passed. ``stream_test`` is the test part with ``cur_dev_s`` estimated
    from GPS: the accuracy to expect on the stream.

    The forecast interval and the classifier metrics come from the rows with the hint the
    model will get: the organisers' one for ``hint="given"``, the GPS one otherwise.

    Returns the final model, the metrics and the out-of-fold predictions. These also hold
    the control model's predictions on the test rows of ``table``, so that :func:`blend` can
    score a blend the same way.
    """
    if config.hint not in HINTS:
        raise ValueError(f"hint must be one of {', '.join(HINTS)}, got {config.hint!r}")
    if _outage(table).any() and config.outage_weight <= 0:
        raise ValueError("the table has outage copies, set outage_weight above 0")
    started = datetime.now(UTC)
    cv = cross_validate(table, config)
    part = table["part"].to_numpy()
    control = fit_models(table[part == "train"], config, cv.iterations, cv.iterations_late)

    oof = table[OOF_COLUMNS + (["hint"] if "hint" in table else [])].copy()
    oof["layover"] = table["new_trip_ahead"].to_numpy() > 0
    oof["fold"] = cv.folds
    oof["base"] = cv.base
    oof["oof"] = cv.oof
    if cv.oof_late is not None:
        oof["oof_late"] = cv.oof_late
    oof["control"] = np.nan
    if (part == "test").any():
        oof.loc[part == "test", "control"] = control.predict(table[part == "test"])["delay_s"]

    hints = hint_rows(table)
    serving = "given" if config.hint == "given" or "gps" not in hints else "gps"
    served = ~oof["synthetic"].to_numpy(dtype=bool) & hints[serving]
    metrics = {
        "cv": cv_scores(oof),
        "iterations": cv.iterations,
        "iterations_late": cv.iterations_late,
        "serving_hint": serving,
    }
    if cv.oof_late is not None:
        y = oof["target"].to_numpy(dtype=float)
        metrics["late"] = classifier_summary(y[served] > LATE_S, cv.oof_late[served])

    if test is None:
        test = table[(part == "test") & hints.get("given", False)]
        if test.empty:
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
        metrics["control"][f"persistence{GPS_SUFFIX}"] = summary(y_stream, gps_cur, gps_cur)
        metrics["control"][f"model{GPS_SUFFIX}"] = summary(
            y_stream, control.predict(stream_test)["delay_s"], gps_cur
        )

    model = fit_models(table, config, cv.iterations, cv.iterations_late)
    model.interval = _interval(oof, served)
    model.meta = _meta(model, started, metrics, config=asdict(config))
    return model, metrics, oof


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def blend(
    members: list[DelayModel], oofs: list[pd.DataFrame], weights: list[float]
) -> tuple[DelayModel, dict, pd.DataFrame]:
    """Weighted average of models trained on the same points with the same folds, as one.

    ``sum w_i * (b_i * two lines + trees_i)`` is one CatBoost model with the trees of all
    members and the two lines with weight ``sum w_i * b_i``. The late classifiers are summed
    in log-odds. Metrics and the interval come from the members' out-of-fold predictions
    combined the same way, so they are as honest as the members' own.

    ``oofs`` are the out-of-fold tables :func:`train` returned for each member.
    """
    if len(members) < 2 or not len(members) == len(oofs) == len(weights):
        raise ValueError("need two or more members, each with its out-of-fold table and weight")
    if min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights must be non-negative and sum to 1, got {weights}")
    first = oofs[0]
    keys = [k for k in ("sample_id", "hint", "fold") if k in first]
    for oof in oofs[1:]:
        same = len(oof) == len(first) and all(
            k in oof and np.array_equal(oof[k].to_numpy(), first[k].to_numpy()) for k in keys
        )
        if not same:
            raise ValueError("members were not trained on the same points with the same folds")
    if any(m.columns != members[0].columns for m in members):
        raise ValueError("members use different feature columns")
    lined = [m for m in members if m.lines is not None]
    if any(m.lines != lined[0].lines for m in lined):
        raise ValueError("members have different two-line baselines")
    serving = {m.meta.get("metrics", {}).get("serving_hint", "given") for m in members}
    if len(serving) > 1:
        raise ValueError(f"members serve different cur_dev_s sources: {sorted(serving)}")
    serving = serving.pop()
    started = datetime.now(UTC)

    classifiers = [m.classifier for m in members]
    model = DelayModel(
        regressor=_average(CatBoostRegressor, [m.regressor for m in members], weights),
        classifier=None
        if None in classifiers
        else _average(CatBoostClassifier, classifiers, weights),
        columns=members[0].columns,
        lines=lined[0].lines if lined else None,
        baseline_weight=sum(
            w * m.baseline_weight
            for w, m in zip(weights, members, strict=True)
            if m.lines is not None
        ),
    )

    oof = first.drop(columns=["oof", "oof_late", "control"], errors="ignore")
    oof["oof"] = sum(w * o["oof"].to_numpy() for w, o in zip(weights, oofs, strict=True))
    if all("control" in o for o in oofs):
        oof["control"] = sum(
            w * o["control"].to_numpy() for w, o in zip(weights, oofs, strict=True)
        )
    served = ~oof["synthetic"].to_numpy(dtype=bool) & hint_rows(oof)[serving]
    metrics = {"cv": cv_scores(oof), "serving_hint": serving}
    if "control" in oof:
        metrics["control"] = control_scores(oof)
    if model.classifier is not None and all("oof_late" in o for o in oofs):
        log_odds = sum(w * _logit(o["oof_late"]) for w, o in zip(weights, oofs, strict=True))
        oof["oof_late"] = 1 / (1 + np.exp(-log_odds))
        y = oof["target"].to_numpy(dtype=float)
        metrics["late"] = classifier_summary(y[served] > LATE_S, oof["oof_late"].to_numpy()[served])

    model.interval = _interval(oof, served)
    model.meta = _meta(
        model,
        started,
        metrics,
        blend={"weights": list(weights), "members": [m.meta.get("config") for m in members]},
    )
    return model, metrics, oof
