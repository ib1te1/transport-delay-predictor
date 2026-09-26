from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from busdelay.cli import main
from busdelay.data import load_part
from busdelay.features import build_features
from busdelay.model import DelayModel, TrainConfig, blend, fit_models, make_matrix, train
from world import write_dataset

SMALL = {"iterations": 40, "depth": 3, "learning_rate": 0.2, "folds": 3}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def parts(dataset):
    return [load_part(dataset, name) for name in ("train", "test")]


@pytest.fixture(scope="module")
def table(parts):
    return pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts], ignore_index=True
    )


@pytest.fixture(scope="module")
def mixed(parts, table):
    gps = pd.concat(
        [build_features(p.points, p.plan, p.telemetry, hint="gps") for p in parts],
        ignore_index=True,
    )
    return pd.concat([table.assign(hint="given"), gps.assign(hint="gps")], ignore_index=True)


def test_train_returns_model_metrics_and_oof(table):
    model, metrics, oof = train(table, TrainConfig(**SMALL))
    assert 1 <= metrics["iterations"] <= 40
    assert 1 <= metrics["iterations_late"] <= 40
    assert set(metrics["cv"]) >= {"persistence", "split_linear", "model"}
    assert metrics["control"]["model"]["rows"] == (table["part"] == "test").sum()
    assert len(oof) == len(table) and oof["oof"].notna().all()
    assert model.interval[0] <= model.interval[1]
    assert model.classifier is not None

    predicted = model.predict(table)
    assert list(predicted.columns) == ["delay_s", "delay_lo_s", "delay_hi_s", "late_prob"]
    assert predicted["late_prob"].between(0, 1).all()


def test_save_and_load_give_the_same_predictions(table, tmp_path):
    model, _, _ = train(table, TrainConfig(**SMALL, vehicle=True))
    loaded = DelayModel.load(model.save(tmp_path / "m"))
    pd.testing.assert_frame_equal(model.predict(table), loaded.predict(table))
    assert loaded.columns[-1] == "vehicle"
    assert loaded.meta["config"]["vehicle"] is True


def test_without_residual_and_classifier(table):
    model, metrics, oof = train(table, TrainConfig(**SMALL, residual=False, classifier=False))
    assert model.lines is None and model.classifier is None
    assert "late" not in metrics and "oof_late" not in oof
    assert np.isfinite(model.predict(table)["delay_s"]).all()


def test_whole_vehicles_held_out(table):
    config = TrainConfig(**{**SMALL, "folds": 2}, group="vehicle", classifier=False)
    _, metrics, oof = train(table, config)
    assert oof.groupby("tr_id")["fold"].nunique().eq(1).all()
    real_folds = oof.loc[~oof["synthetic"], "fold"]
    assert set(real_folds) == {0, 1}
    assert metrics["cv"]["model"]["rows"] == (~table["synthetic"]).sum()


def test_no_late_points_means_no_classifier(table):
    calm = table.assign(target=table["target"].clip(upper=60))
    model, metrics, _ = train(calm, TrainConfig(**SMALL))
    assert model.classifier is None and "late" not in metrics
    assert "late_prob" not in model.predict(calm)


def test_seeds_are_averaged_into_one_model(table):
    config = TrainConfig(**SMALL)
    single = [fit_models(table, replace(config, seed=s), 20, 20) for s in (0, 1)]
    merged = fit_models(table, replace(config, seeds=2), 20, 20)

    expected = np.mean([m.predict(table)["delay_s"] for m in single], axis=0)
    np.testing.assert_allclose(merged.predict(table)["delay_s"], expected, atol=1e-6)
    X = make_matrix(table, merged.columns)
    log_odds = np.mean([m.classifier.predict(X, "RawFormulaVal") for m in single], axis=0)
    np.testing.assert_allclose(
        merged.predict(table)["late_prob"], 1 / (1 + np.exp(-log_odds)), atol=1e-6
    )


def test_mixed_hints_are_scored_separately(mixed):
    _, metrics, oof = train(mixed, TrainConfig(**SMALL, hint="mix"))
    assert {"model", "model, cur_dev from GPS"} <= set(metrics["cv"])
    assert metrics["serving_hint"] == "gps"
    real_given = (~mixed["synthetic"] & (mixed["hint"] == "given")).sum()
    assert metrics["cv"]["model"]["rows"] == real_given
    # both copies of a point are held out together
    assert oof.groupby("sample_id")["fold"].nunique().eq(1).all()
    test_given = (mixed["part"] == "test") & (mixed["hint"] == "given")
    assert metrics["control"]["model"]["rows"] == test_given.sum()


def test_blend_is_the_weighted_average_of_its_members(mixed, tmp_path):
    residual = train(mixed, TrainConfig(**SMALL, hint="mix"))
    plain = train(mixed, TrainConfig(**{**SMALL, "depth": 4}, hint="mix", residual=False))
    model, metrics, oof = blend([residual[0], plain[0]], [residual[2], plain[2]], [0.7, 0.3])

    assert model.baseline_weight == pytest.approx(0.7)
    a, b = residual[0].predict(mixed), plain[0].predict(mixed)
    got = model.predict(mixed)
    np.testing.assert_allclose(got["delay_s"], 0.7 * a["delay_s"] + 0.3 * b["delay_s"], atol=1e-6)
    log_odds = 0.7 * np.log(a["late_prob"] / (1 - a["late_prob"])) + 0.3 * np.log(
        b["late_prob"] / (1 - b["late_prob"])
    )
    np.testing.assert_allclose(got["late_prob"], 1 / (1 + np.exp(-log_odds)), atol=1e-6)

    expected_oof = 0.7 * residual[2]["oof"] + 0.3 * plain[2]["oof"]
    np.testing.assert_allclose(oof["oof"], expected_oof)
    assert {"model", "model, cur_dev from GPS"} <= set(metrics["cv"])
    assert {"model", "model, cur_dev from GPS"} <= set(metrics["control"])
    assert metrics["serving_hint"] == "gps" and "late" in metrics

    loaded = DelayModel.load(model.save(tmp_path / "blend"))
    pd.testing.assert_frame_equal(loaded.predict(mixed), got)
    assert loaded.meta["blend"]["weights"] == [0.7, 0.3]


def test_blend_refuses_members_with_other_folds(table):
    one = train(table, TrainConfig(**SMALL))
    other = train(table, TrainConfig(**SMALL, seed=1))
    with pytest.raises(ValueError, match="same folds"):
        blend([one[0], other[0]], [one[2], other[2]], [0.5, 0.5])
    with pytest.raises(ValueError, match="sum to 1"):
        blend([one[0], one[0]], [one[2], one[2]], [0.5, 0.6])


def test_unknown_hint_is_refused(table):
    with pytest.raises(ValueError, match="hint"):
        train(table, TrainConfig(**SMALL, hint="guess"))


def test_matrix_keeps_the_column_order(table):
    columns = ["cur_dev_s", "lead_s", "vehicle"]
    X = make_matrix(table.iloc[:, ::-1], columns)
    assert list(X.columns) == columns
    assert X["vehicle"].map(type).eq(str).all()


def test_train_and_predict_commands(dataset, tmp_path, capsys):
    features = tmp_path / "features"
    models = tmp_path / "models"
    oof_dir = tmp_path / "oof"
    out = tmp_path / "sub.csv"
    assert main(["features", "--data", str(dataset), "--out", str(features)]) == 0
    train_args = ["train", "--name", "tiny", "--features", str(features), "--models", str(models)]
    train_args += ["--oof", str(oof_dir), "--iterations", "30", "--depth", "3", "--folds", "3"]
    assert main(train_args) == 0
    assert (oof_dir / "tiny.parquet").exists()
    assert not (models / "tiny" / "oof.parquet").exists()
    predict_args = ["predict", "--model", str(models / "tiny"), "--data", str(dataset)]
    predict_args += ["--out", str(out)]
    assert main(predict_args) == 0
    assert "ok" in capsys.readouterr().out
    # the service's path gives what the feature table gives
    written = pd.read_csv(out, sep=";", dtype={"sample_id": str})
    validate = pd.read_parquet(features / "validate.parquet")
    offline = DelayModel.load(models / "tiny").predict(validate)["delay_s"]
    assert written["sample_id"].tolist() == validate["sample_id"].tolist()
    np.testing.assert_allclose(written["prediction"], np.round(offline, 1))

    bench_args = ["bench", "--model", str(models / "tiny"), "--data", str(dataset)]
    assert main([*bench_args, "--moments", "3"]) == 0
    printed = capsys.readouterr().out
    assert "one vehicle alone" in printed and "fell back" not in printed

    mix_args = [*train_args, "--hint", "mix", "--seeds", "2"]
    mix_args[mix_args.index("tiny")] = "mix"
    assert main(mix_args) == 0
    oof = pd.read_parquet(oof_dir / "mix.parquet")
    assert not oof["synthetic"].any()
    assert set(oof["hint"]) == {"given", "gps"}
    meta = DelayModel.load(models / "mix").meta
    assert meta["config"]["hint"] == "mix" and meta["config"]["seeds"] == 2

    plain_args = [*mix_args, "--no-residual"]
    plain_args[plain_args.index("mix")] = "plain"
    assert main(plain_args) == 0
    blend_args = ["blend", "--name", "both", "--models", str(models), "--oof", str(oof_dir)]
    blend_args += ["--model", str(models / "mix"), str(models / "plain")]
    assert main(blend_args) == 0
    meta = DelayModel.load(models / "both").meta
    assert meta["blend"]["names"] == ["mix", "plain"]
    assert meta["metrics"]["serving_hint"] == "gps"
    assert (oof_dir / "both.parquet").exists()
