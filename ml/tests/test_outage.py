import numpy as np
import pandas as pd
import pytest

from busdelay.cli import main
from busdelay.data import load_part
from busdelay.features import build_features
from busdelay.model import TrainConfig, train
from busdelay.outage import GAP_S, KINDS, evaluate, outage_points
from world import write_dataset

SMALL = {"iterations": 40, "depth": 3, "learning_rate": 0.2, "folds": 3}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def parts(dataset):
    return {name: load_part(dataset, name) for name in ("train", "test")}


@pytest.fixture(scope="module")
def model(parts):
    table = pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts.values()],
        ignore_index=True,
    )
    return train(table, TrainConfig(**SMALL))[0]


def test_outage_points_are_copies_of_the_real_points(parts):
    points = parts["train"].points
    real = points[points["tr_id"] < 9_000_000]
    copies = outage_points(points, copies=2, seed=3)

    assert len(copies) == 2 * len(real)
    assert set(copies["sample_id"]) == set(real["sample_id"])
    assert copies["gap_s"].between(*GAP_S).all()
    assert set(copies["outage"]) == set(KINDS)
    pd.testing.assert_frame_equal(copies, outage_points(points, copies=2, seed=3))


def test_evaluate_scores_every_outage(parts, model):
    table = evaluate({"a": model, "b": model}, parts["test"], minutes=[0, 5])

    assert list(zip(table["outage"], table["minutes"], strict=True)) == [
        ("none", 0),
        ("silence", 5),
        ("no_position", 5),
    ]
    assert {"persistence", "zero", "split_linear", "a", "b"} <= set(table.columns)
    # the same model twice: no difference at all
    assert (table["- a"] == 0).all() and (table["- a lo"] == 0).all()
    np.testing.assert_allclose(table["- split_linear"], table["b"] - table["split_linear"])
    assert (table["- split_linear lo"] <= table["- split_linear"]).all()

    fresh = evaluate({"a": model}, parts["test"], minutes=[0])
    test = parts["test"]
    features = build_features(test.points, test.plan, test.telemetry, hint="gps")
    error = (features["target"] - model.predict(features)["delay_s"]).abs()
    assert fresh["a"].iloc[0] == pytest.approx(error[features["tel_age_s"] <= 30].mean())
    assert "- split_linear" in fresh and "- a" not in fresh


def test_outage_copies_have_their_own_metrics_and_leave_the_lines_alone(parts):
    frames = []
    for p in parts.values():
        frames.append(build_features(p.points, p.plan, p.telemetry).assign(hint="given"))
        frames.append(build_features(p.points, p.plan, p.telemetry, hint="gps").assign(hint="gps"))
    mixed = pd.concat(frames, ignore_index=True)
    copies = pd.concat(
        [
            build_features(outage_points(p.points), p.plan, p.telemetry, hint="gps")
            for p in parts.values()
        ],
        ignore_index=True,
    )
    copies = copies.drop(columns=["gap_s", "outage"]).assign(hint="outage")
    table = pd.concat([mixed, copies], ignore_index=True)

    config = TrainConfig(**SMALL, hint="mix", outage_weight=0.5)
    model, metrics, oof = train(table, config)
    without = train(mixed, TrainConfig(**SMALL, hint="mix"))[0]

    assert "model, telemetry cut 1-15 min" in metrics["cv"]
    assert model.lines == without.lines
    assert metrics["serving_hint"] == "gps"
    assert oof.groupby(["sample_id"])["fold"].nunique().eq(1).all()
    with pytest.raises(ValueError, match="outage_weight"):
        train(table, TrainConfig(**SMALL, hint="mix"))


def test_features_train_and_outage_commands(dataset, tmp_path, capsys):
    features, models = tmp_path / "features", tmp_path / "models"
    assert main(["features", "--data", str(dataset), "--out", str(features)]) == 0
    assert not (features / "train_outage.parquet").exists()
    assert main(["features", "--data", str(dataset), "--out", str(features), "--outage"]) == 0
    for part in ("train", "test"):
        copies = pd.read_parquet(features / f"{part}_outage.parquet")
        assert copies["gap_s"].between(*GAP_S).all() and not copies["synthetic"].any()
    assert not (features / "validate_outage.parquet").exists()

    common = ["--features", str(features), "--models", str(models), "--oof", str(tmp_path)]
    common += ["--iterations", "30", "--depth", "3", "--folds", "3", "--seeds", "1"]
    common += ["--parts", "train", "--hint", "mix"]
    assert main(["train", "--name", "plain", *common]) == 0
    assert main(["train", "--name", "cut", *common, "--outage-weight", "0.5"]) == 0
    assert set(pd.read_parquet(tmp_path / "cut.parquet")["hint"]) == {"given", "gps", "outage"}

    capsys.readouterr()
    args = ["outage", "--model", str(models / "plain"), str(models / "cut")]
    assert main([*args, "--data", str(dataset), "--minutes", "0", "3"]) == 0
    printed = capsys.readouterr().out
    assert "no_position" in printed and "silence" in printed
    assert "cut - split_linear" in printed and "cut - plain" in printed
    assert np.isfinite(pd.read_parquet(tmp_path / "cut.parquet")["oof"]).all()
