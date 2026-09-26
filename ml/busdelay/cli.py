"""Command line entry point, ``python -m busdelay <command>``.

Commands:

* ``features`` - build feature tables for train, test and validate
* ``arrivals`` - check the GPS arrival detector against the schedule facts
* ``baselines`` - cross-validate the simple predictors
* ``submit-baseline`` - fit one baseline on all labeled points and write a submission
* ``train`` - cross-validate CatBoost, then fit it on all labeled points and save it
* ``blend`` - average models trained on the same points into one model
* ``predict`` - predict validate the way the service does and write a submission
* ``bench`` - time the service's forecaster on validate telemetry
* ``outage`` - MAE on test with the last minutes of telemetry cut
* ``check`` - check a submission file against ``validate/points.csv``

Paths default to the layout of the repository when run from ``ml/``.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .arrivals import detect_arrivals
from .baselines import BASELINES
from .data import FILES, PARTS, is_synthetic, load_part, read_points
from .features import HISTORY_S, build_features, read_feature_table
from .folds import block_ids, group_folds
from .inference import FIX_COLUMNS, Forecaster, Query, queries_for_points
from .metrics import summary
from .model import HINTS, DelayModel, TrainConfig, blend, train
from .outage import KINDS, evaluate, outage_points
from .schedule import VehiclePlan, prepare_plan, split_by_vehicle
from .submission import check_submission, write_submission

DEFAULT_DATA = Path("../data/dataset")
DEFAULT_FEATURES = Path("../data/features")
DEFAULT_MODELS = Path("../models")
# out-of-fold predictions hold the labels, so they stay in data/ and not next to the model
DEFAULT_OOF = Path("../data/oof")
DEFAULT_SUBMISSIONS = Path("../data/submissions")


def cmd_features(args) -> int:
    """``features``: feature tables, and the same points with ``cur_dev_s`` from GPS."""
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.parts:
        started = time.perf_counter()
        part = load_part(args.data, name)
        table = build_features(part.points, part.plan, part.telemetry)
        path = args.out / f"{name}.parquet"
        table.to_parquet(path, index=False)
        print(f"{name}: {len(table)} points in {time.perf_counter() - started:.1f} s -> {path}")
        if name != "validate":
            # the same points with cur_dev_s estimated from GPS, as on the live stream
            stream = build_features(part.points, part.plan, part.telemetry, hint="gps")
            stream.to_parquet(args.out / f"{name}_stream.parquet", index=False)
            print(f"{name}_stream: {len(stream)} points, cur_dev_s from GPS")
        if name != "validate" and args.outage:
            # copies of the real points with the position lost before T, for --outage-weight
            copies = outage_points(part.points, seed=PARTS.index(name))
            outage = build_features(copies, part.plan, part.telemetry, hint="gps")
            outage.to_parquet(args.out / f"{name}_outage.parquet", index=False)
            print(f"{name}_outage: {len(outage)} copies with the telemetry cut")
    return 0


def read_labeled(features: Path, parts: list[str], hint: str, outage: bool = False) -> pd.DataFrame:
    """Labeled points with the organisers' ``cur_dev_s``, the GPS one, or both (``mix``).

    With ``outage`` the outage copies are added with hint ``outage``.
    """
    frames = []
    if hint in ("given", "mix"):
        frames.append(read_feature_table(features, parts).assign(hint="given"))
    if hint in ("gps", "mix"):
        stream = read_feature_table(features, [f"{part}_stream" for part in parts])
        frames.append(stream.assign(hint="gps"))
    if outage:
        copies = read_feature_table(features, [f"{part}_outage" for part in parts])
        frames.append(copies.drop(columns=["gap_s", "outage"]).assign(hint="outage"))
    return pd.concat(frames, ignore_index=True)


def cmd_arrivals(args) -> int:
    """``arrivals``: how close the GPS arrival detector gets to the schedule facts."""
    for name in ("train", "test"):
        part = load_part(args.data, name)
        detected = []
        telemetry = part.telemetry[part.telemetry["ok"]]
        for tr_id, plan in split_by_vehicle(prepare_plan(part.plan)).items():
            fixes = telemetry[telemetry["tr_id"] == tr_id]
            trip_start = plan.seq == 0
            arrived, _, _ = detect_arrivals(
                fixes["t"].to_numpy(),
                fixes["lat"].to_numpy(),
                fixes["lon"].to_numpy(),
                plan.t_plan,
                plan.lat,
                plan.lon,
                departs=trip_start,
            )
            detected.append(
                pd.DataFrame({"stop_id": plan.stop_id, "t_arr": arrived, "trip_start": trip_start})
            )
        table = pd.concat(detected).merge(part.facts, on="stop_id")
        vehicle = part.plan.set_index("stop_id").loc[table["stop_id"], "tr_id"]
        table["synthetic"] = is_synthetic(vehicle)
        table["error"] = table["t_arr"] - table["t_fact"]

        print(f"\n{name}: detected time minus fact time")
        print(
            f"{'stops':>36} {'rows':>6} {'found':>6} {'<=15s':>6} {'<=30s':>6} {'<=60s':>6} "
            f"{'med|e|':>7}"
        )
        groups = table.groupby(["synthetic", "manual_fill", "trip_start"])
        for (synthetic, manual, start), rows in groups:
            error = rows["error"].abs()
            label = (
                f"{'synthetic' if synthetic else 'real'}, {'manual' if manual else 'gps'}, "
                f"{'trip start' if start else 'other'}"
            )
            print(
                f"{label:>36} {len(rows):6d} {rows['t_arr'].notna().mean():6.1%} "
                f"{(error <= 15).mean():6.1%} {(error <= 30).mean():6.1%} "
                f"{(error <= 60).mean():6.1%} {error.median():7.1f}"
            )
    return 0


def cross_validate(table: pd.DataFrame, make_model, n_folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions for every row of ``table``."""
    folds = group_folds(block_ids(table), n_folds=n_folds, seed=seed)
    y = table["target"].to_numpy(dtype=float)
    predictions = np.zeros(len(table))
    for fold in range(n_folds):
        train = folds != fold
        model = make_model().fit(table[train], y[train])
        predictions[~train] = model.predict(table[~train])
    return predictions


def print_scores(rows: dict[str, dict[str, float]]) -> None:
    """Print a table of :func:`busdelay.metrics.summary` rows."""
    print(f"{'':>36} {'MAE':>7} {'vs pers':>8} {'bias':>7} {'p90':>7} {'score~':>7} {'rows':>6}")
    for name, s in rows.items():
        print(
            f"{name:>36} {s['mae']:7.1f} {s['vs_persistence']:+8.1%} {s['bias']:+7.1f} "
            f"{s['p90']:7.1f} {s['score']:7.2f} {s['rows']:6d}"
        )


def cmd_baselines(args) -> int:
    """``baselines``: cross-validate the simple predictors on the same folds as the model."""
    table = read_feature_table(args.features, ["train", "test"])
    y = table["target"].to_numpy(dtype=float)
    cur = table["cur_dev_s"].fillna(0.0).to_numpy()
    real = ~table["synthetic"].to_numpy()
    layover = table["new_trip_ahead"].to_numpy() > 0

    cv, control = {}, {}
    for name, cls in BASELINES.items():
        oof = cross_validate(table, cls, args.folds, args.seed)
        cv[name] = summary(y[real], oof[real], cur[real])
        cv[f"{name} / layover"] = summary(
            y[real & layover], oof[real & layover], cur[real & layover]
        )

        train = table["part"].to_numpy() == "train"
        test = ~train
        model = cls().fit(table[train], y[train])
        control[name] = summary(y[test], model.predict(table[test]), cur[test])

    print(f"{args.folds}-fold CV over train+test, scored on real vehicles:")
    print_scores(cv)
    print("\nfit on train, scored on test (the organisers' split):")
    print_scores(control)
    print("\nscore~ is a rough estimate of the platform score, see busdelay/metrics.py")
    return 0


def cmd_submit_baseline(args) -> int:
    """``submit-baseline``: fit one baseline on all labeled points and write a submission."""
    labeled = read_feature_table(args.features, ["train", "test"])
    validate = read_feature_table(args.features, ["validate"])
    model = BASELINES[args.kind]().fit(labeled, labeled["target"].to_numpy(dtype=float))
    out = args.out or DEFAULT_SUBMISSIONS / f"{args.kind}.csv"
    path = write_submission(validate["sample_id"], model.predict(validate), out)
    return report_check(path, args.data)


def cmd_train(args) -> int:
    """``train``: cross-validate CatBoost, then fit it on all labeled points and save it."""
    config = TrainConfig(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        residual=not args.no_residual,
        vehicle=args.vehicle,
        synthetic_weight=args.synthetic_weight,
        outage_weight=args.outage_weight,
        classifier=not args.no_classifier,
        folds=args.folds,
        group=args.group,
        gpu=args.gpu,
        seed=args.seed,
        seeds=args.seeds,
        hint=args.hint,
    )
    table = read_labeled(args.features, args.parts, args.hint, outage=args.outage_weight > 0)
    if not args.with_synthetic:
        table = table[~table["synthetic"]].reset_index(drop=True)
    test = read_feature_table(args.features, ["test"])
    stream_path = args.features / "test_stream.parquet"
    stream = read_feature_table(args.features, ["test_stream"]) if stream_path.exists() else None
    print(f"training {args.name} on {'+'.join(args.parts)}: {len(table)} points, {config}")
    started = time.perf_counter()
    model, metrics, oof = train(table, config, stream_test=stream, test=test)
    out = save_model(model, oof, args)
    print(f"\n{config.folds}-fold CV, real vehicles, {metrics['iterations']} trees:")
    print_report(model, metrics)
    print(f"\n{time.perf_counter() - started:.0f} s, saved to {out}")
    print(f"next: python -m busdelay predict --model {out}")
    return 0


def cmd_blend(args) -> int:
    """``blend``: average models trained on the same points into one saved model."""
    weights = args.weights or [1 / len(args.model)] * len(args.model)
    if len(weights) != len(args.model):
        raise ValueError(f"{len(args.model)} models but {len(weights)} weights")
    members = [DelayModel.load(path) for path in args.model]
    oofs = [pd.read_parquet(args.oof / f"{Path(path).name}.parquet") for path in args.model]
    model, metrics, oof = blend(members, oofs, weights)
    model.meta["blend"]["names"] = [Path(path).name for path in args.model]
    out = save_model(model, oof, args)
    names = ", ".join(f"{Path(p).name} x{w:g}" for p, w in zip(args.model, weights, strict=True))
    print(f"blend of {names}, out-of-fold predictions of the members, real vehicles:")
    print_report(model, metrics)
    print(f"\nsaved to {out}")
    return 0


def save_model(model: DelayModel, oof: pd.DataFrame, args) -> Path:
    """The model to models/<name>, its out-of-fold predictions to data/oof/<name>.parquet."""
    out = model.save(args.models / args.name)
    args.oof.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof / f"{args.name}.parquet", index=False)
    return out


def print_report(model: DelayModel, metrics: dict) -> None:
    """Print the CV and control metrics a model was saved with."""
    print_scores(metrics["cv"])
    if "control" in metrics:
        print("\nfit on train, scored on test (the organisers' split):")
        print_scores(metrics["control"])
    if "late" in metrics:
        late = metrics["late"]
        trees = f", {metrics['iterations_late']} trees" if "iterations_late" in metrics else ""
        print(
            f"\nlate > 2 min classifier{trees}: "
            f"AUC {late['auc']:.3f}, logloss {late['logloss']:.3f}, "
            f"Brier {late['brier']:.3f}, share late {late['positive_rate']:.1%}"
        )
    lo, hi = model.interval
    print(f"interval (10%..90% of residuals): {lo:+.0f} .. {hi:+.0f} s")
    print("\nmost important features:")
    for name, value in model.importance(15):
        print(f"  {name:>20} {value:6.2f}")


def cmd_predict(args) -> int:
    """``predict``: write a submission for validate the way the service predicts."""
    # the service's path: a query per point, then the forecaster, with the organisers'
    # cur_dev_s that validate has
    part = load_part(args.data, "validate")
    queries = queries_for_points(part.points, part.plan, part.telemetry)
    runs = [Forecaster(DelayModel.load(path), hint="given").predict(queries) for path in args.model]
    failed = [a for a in runs[0] if a.fallback is not None]
    if failed:
        print(
            f"{len(failed)} points got cur_dev_s instead of the model, first: {failed[0].fallback}"
        )
    # several models are averaged
    delay = np.mean([[a.delay_s for a in answers] for answers in runs], axis=0)
    name = "+".join(Path(path).name for path in args.model)
    out = args.out or DEFAULT_SUBMISSIONS / f"{name}.csv"
    path = write_submission([a.sample_id for a in runs[0]], delay, out)
    print(f"{len(delay)} predictions: mean {delay.mean():+.0f} s, median {np.median(delay):+.0f} s")
    return report_check(path, args.data)


def bench_queries(
    plans: dict[int, VehiclePlan], fixes: dict[int, pd.DataFrame], now: float
) -> list[Query]:
    """A query per vehicle that has a stop 10-15 minutes ahead of ``now``, the way the api
    asks: the plan from the start of the day up to the target, no ``cur_dev_s`` and
    :data:`busdelay.features.HISTORY_S` of telemetry before ``now``."""
    empty = pd.DataFrame(columns=FIX_COLUMNS)
    queries = []
    for tr_id, plan in plans.items():
        target = plan.pick_target(now)
        if target < 0:
            continue
        stops = pd.DataFrame(
            {"stop_id": plan.stop_id, "t_plan": plan.t_plan, "lat": plan.lat, "lon": plan.lon}
        )
        track = fixes.get(tr_id, empty)
        queries.append(
            Query(
                sample_id=f"{tr_id}_{now:.0f}",
                tr_id=tr_id,
                T=float(now),
                target_stop_id=int(plan.stop_id[target]),
                cur_dev_s=np.nan,
                plan=stops.iloc[: target + 1],
                fixes=track[(track["t"] >= now - HISTORY_S) & (track["t"] <= now)],
            )
        )
    return queries


def cmd_bench(args) -> int:
    """``bench``: time the service's :class:`busdelay.inference.Forecaster` on validate."""
    forecaster = Forecaster(DelayModel.load(args.model), hint="gps")
    part = load_part(args.data, "validate")
    plans = split_by_vehicle(prepare_plan(part.plan))
    fixes = {
        int(tr_id): rows.rename(columns={"ok": "location_valid"})
        for tr_id, rows in part.telemetry.groupby("tr_id")
    }
    moments = np.linspace(part.points["T"].min(), part.points["T"].max(), args.moments)

    single, cycles, forecasts, fallbacks = [], [], 0, 0
    for now in moments:
        queries = bench_queries(plans, fixes, now)
        if not queries:
            continue
        started = time.perf_counter()
        answers = forecaster.predict(queries)
        cycles.append((time.perf_counter() - started) * 1000)
        forecasts += len(answers)
        fallbacks += sum(a.fallback is not None for a in answers)
        for query in queries:
            started = time.perf_counter()
            forecaster.predict([query])
            single.append((time.perf_counter() - started) * 1000)

    if not forecasts:
        raise ValueError("no vehicle has a stop 10-15 minutes ahead at any moment")
    single, cycles = np.array(single), np.array(cycles)
    print(f"{len(plans)} vehicles, {len(moments)} moments, {forecasts} forecasts")
    if fallbacks:
        print(f"{fallbacks} forecasts fell back to cur_dev_s, the timings are too low")
    print(
        f"all vehicles in one call: p50 {np.percentile(cycles, 50):.0f} ms, "
        f"max {cycles.max():.0f} ms, {cycles.sum() / forecasts:.1f} ms per vehicle"
    )
    print(
        f"one vehicle alone: p50 {np.percentile(single, 50):.0f} ms, "
        f"p95 {np.percentile(single, 95):.0f} ms"
    )
    return 0


def cmd_outage(args) -> int:
    """``outage``: MAE on test with the telemetry cut before ``T``, see :mod:`busdelay.outage`."""
    models = {Path(path).name: DelayModel.load(path) for path in args.model}
    if len(models) != len(args.model):
        raise ValueError("models must have different directory names")
    part = load_part(args.data, "test")
    table = evaluate(models, part, minutes=args.minutes, kinds=args.kinds)
    print(
        f"test, {table['points'].iloc[0]} real points with fresh telemetry, "
        "cur_dev_s from GPS, MAE:"
    )
    maes = [c for c in table.columns[3:] if not c.startswith("- ")]
    differences = [c for c in table.columns if c.startswith("- ") and c[-3:] not in (" lo", " hi")]
    last = list(models)[-1]
    header = [f"{n[:12]:>12}" for n in maes] + [
        f"{last[:10]} {d}"[:22].rjust(22) for d in differences
    ]
    print(f"{'outage':>12} {'min':>4} " + " ".join(header))
    for row in table.to_dict("records"):
        cells = [f"{row[n]:12.1f}" for n in maes]
        cells += [
            f"{row[d]:+.1f} [{row[d + ' lo']:+.1f}, {row[d + ' hi']:+.1f}]".rjust(22)
            for d in differences
        ]
        print(f"{row['outage']:>12} {row['minutes']:4g} " + " ".join(cells))
    print("differences: MAE, 90% bootstrap interval over runs of points")
    print("honest only for models trained without test (train --parts train)")
    return 0


def cmd_check(args) -> int:
    """``check``: check a submission file."""
    return report_check(args.file, args.data)


def report_check(path: Path, data: Path) -> int:
    """Check a submission against ``validate/points.csv`` and print the result; 0 if it is fine."""
    expected = read_points(data / FILES["validate"][2], part="validate")["sample_id"]
    problems = check_submission(path, expected)
    if problems:
        print(f"{path}: NOT OK")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{path}: ok, {len(expected)} rows")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The parser with every command and its options."""
    parser = argparse.ArgumentParser(prog="busdelay", description="Bus delay model.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("features", help="build feature tables")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="unpacked dataset")
    p.add_argument("--out", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    p.add_argument(
        "--outage", action="store_true", help="also copies with the telemetry cut before T"
    )
    p.set_defaults(func=cmd_features)

    p = commands.add_parser("arrivals", help="check detected arrivals against facts")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.set_defaults(func=cmd_arrivals)

    p = commands.add_parser("baselines", help="cross-validate the baselines")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_baselines)

    p = commands.add_parser("submit-baseline", help="write a submission from a baseline")
    p.add_argument("kind", choices=sorted(BASELINES))
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=None, help="default: data/submissions/<kind>.csv")
    p.set_defaults(func=cmd_submit_baseline)

    p = commands.add_parser("train", help="cross-validate and train the CatBoost model")
    p.add_argument("--name", required=True, help="the model is saved to models/<name>")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--oof", type=Path, default=DEFAULT_OOF, help="out-of-fold predictions")
    p.add_argument("--iterations", type=int, default=3000, help="max trees, the CV picks fewer")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=30.0)
    p.add_argument("--no-residual", action="store_true", help="learn the delay, not the residual")
    p.add_argument("--vehicle", action="store_true", help="add tr_id as a category")
    p.add_argument("--synthetic-weight", type=float, default=1.0)
    p.add_argument(
        "--outage-weight",
        type=float,
        default=0.0,
        help="add the copies from features --outage with this weight; 0 - without",
    )
    p.add_argument("--no-classifier", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--group",
        choices=["runs", "vehicle", "route"],
        default="runs",
        help="runs: like the organisers' split; vehicle: whole vehicles held out; "
        "route: a real vehicle with its synthetic copies held out",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5, help="seeds averaged in the final model")
    p.add_argument("--gpu", action="store_true", help="train on the GPU (CUDA)")
    p.add_argument(
        "--parts",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="labeled parts to train on; with train only, test is just the control",
    )
    p.add_argument(
        "--hint",
        choices=HINTS,
        default="given",
        help="cur_dev_s to train on: given - the organisers' (for the submission), "
        "gps - estimated from GPS as on the stream, mix - both (for the service)",
    )
    p.add_argument(
        "--with-synthetic",
        action="store_true",
        help="keep the synthetic copies of real vehicles; off by default, the model then "
        "finds the copy and takes its delay (docs/specs/ml-model.md)",
    )
    p.set_defaults(func=cmd_train)

    p = commands.add_parser("blend", help="average trained models into one")
    p.add_argument("--name", required=True, help="the blend is saved to models/<name>")
    p.add_argument(
        "--model",
        type=Path,
        nargs="+",
        required=True,
        help="models/<name> trained on the same points with the same folds",
    )
    p.add_argument("--weights", type=float, nargs="+", default=None, help="default: equal")
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--oof", type=Path, default=DEFAULT_OOF, help="out-of-fold predictions")
    p.set_defaults(func=cmd_blend)

    p = commands.add_parser("predict", help="write a submission with saved models")
    p.add_argument(
        "--model", type=Path, nargs="+", required=True, help="models/<name>, several are averaged"
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=None, help="default: data/submissions/<name>.csv")
    p.set_defaults(func=cmd_predict)

    p = commands.add_parser("bench", help="time the forecaster on validate telemetry")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--moments", type=int, default=20)
    p.set_defaults(func=cmd_bench)

    p = commands.add_parser("outage", help="MAE on test with the telemetry cut before T")
    p.add_argument("--model", type=Path, nargs="+", required=True, help="models/<name>")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--minutes", type=float, nargs="+", default=[0, 2, 5, 10, 15], help="outage lengths"
    )
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    p.set_defaults(func=cmd_outage)

    p = commands.add_parser("check", help="check a submission file")
    p.add_argument("file", type=Path)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command; returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"busdelay: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
