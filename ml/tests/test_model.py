import numpy as np
import pandas as pd
import pytest

from busdelay.cli import main
from busdelay.data import load_part
from busdelay.features import build_features
from busdelay.model import DelayModel, TrainConfig, make_matrix, train
from world import write_dataset

SMALL = {"iterations": 40, "depth": 3, "learning_rate": 0.2, "folds": 3}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def table(dataset):
    parts = [load_part(dataset, name) for name in ("train", "test")]
    return pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts], ignore_index=True
    )


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


def test_matrix_keeps_the_column_order(table):
    columns = ["cur_dev_s", "lead_s", "vehicle"]
    X = make_matrix(table.iloc[:, ::-1], columns)
    assert list(X.columns) == columns
    assert X["vehicle"].map(type).eq(str).all()


def test_train_and_predict_commands(dataset, tmp_path, capsys):
    features = tmp_path / "features"
    models = tmp_path / "models"
    out = tmp_path / "sub.csv"
    assert main(["features", "--data", str(dataset), "--out", str(features)]) == 0
    train_args = ["train", "--name", "tiny", "--features", str(features), "--models", str(models)]
    train_args += ["--iterations", "30", "--depth", "3", "--folds", "3"]
    assert main(train_args) == 0
    assert (models / "tiny" / "oof.parquet").exists()
    predict_args = ["predict", "--model", str(models / "tiny"), "--features", str(features)]
    predict_args += ["--data", str(dataset), "--out", str(out)]
    assert main(predict_args) == 0
    assert "ok" in capsys.readouterr().out
