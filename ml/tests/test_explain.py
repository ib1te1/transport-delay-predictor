import numpy as np
import pandas as pd
import pytest

from busdelay.data import load_part
from busdelay.explain import REASONS, main_reason, reason_contributions
from busdelay.features import FEATURES, build_features
from busdelay.model import TrainConfig, blend, train
from world import write_dataset

SMALL = {"iterations": 40, "depth": 3, "learning_rate": 0.2, "folds": 3, "classifier": False}


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    data = write_dataset(tmp_path_factory.mktemp("raw"))
    parts = [load_part(data, name) for name in ("train", "test")]
    return pd.concat(
        [build_features(p.points, p.plan, p.telemetry) for p in parts], ignore_index=True
    )


def test_every_feature_has_exactly_one_reason():
    grouped = [f for _, features in REASONS.values() for f in features if f != "vehicle"]
    assert sorted(grouped) == sorted(FEATURES)


@pytest.mark.parametrize("kind", ["residual", "plain", "blend"])
def test_reasons_add_up_to_the_prediction(table, kind):
    if kind == "blend":
        members = [train(table, TrainConfig(**SMALL, residual=r)) for r in (True, False)]
        model, _, _ = blend([m for m, _, _ in members], [o for _, _, o in members], [0.5, 0.5])
        assert model.baseline_weight == 0.5
    else:
        model, _, _ = train(table, TrainConfig(**SMALL, residual=kind == "residual"))
    parts = reason_contributions(model, table)
    total = parts[list(REASONS)].sum(axis=1) + parts["expected_s"]
    np.testing.assert_allclose(total, model.predict(table)["delay_s"], atol=1e-6)


def test_main_reason_picks_the_biggest_push():
    parts = pd.DataFrame({key: [0.0, 0.0] for key in REASONS})
    parts.loc[0, "long_dwell"] = 90.0
    parts.loc[0, "layover"] = -200.0
    parts.loc[1, "layover"] = -150.0
    parts.loc[1, "signal"] = -10.0
    reasons = main_reason(parts)
    assert reasons["reason"].tolist() == ["long_dwell", "layover"]
    assert reasons["reason_s"].tolist() == [90.0, -150.0]
    assert reasons["reason_text"].iloc[0] == "Долгая стоянка или посадка"
