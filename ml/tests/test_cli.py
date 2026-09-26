import pandas as pd

from busdelay.cli import main
from world import write_dataset


def test_features_baselines_and_submission_on_a_fake_dataset(tmp_path, capsys):
    data = write_dataset(tmp_path / "raw")
    features = tmp_path / "features"

    assert main(["features", "--data", str(data), "--out", str(features)]) == 0
    for part in ("train", "test", "validate"):
        assert (features / f"{part}.parquet").exists()
    validate = pd.read_parquet(features / "validate.parquet")
    assert validate["target"].isna().all()
    assert not validate["synthetic"].any()

    assert main(["arrivals", "--data", str(data)]) == 0
    assert main(["baselines", "--features", str(features), "--folds", "3"]) == 0
    printed = capsys.readouterr().out
    assert "persistence" in printed and "split_linear" in printed

    out = tmp_path / "sub.csv"
    code = main(
        [
            "submit-baseline",
            "split_linear",
            "--features",
            str(features),
            "--data",
            str(data),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert main(["check", str(out), "--data", str(data)]) == 0


def test_check_fails_on_a_broken_file(tmp_path):
    data = write_dataset(tmp_path / "raw")
    broken = tmp_path / "broken.csv"
    broken.write_text("sample_id;prediction\n", encoding="utf-8")
    assert main(["check", str(broken), "--data", str(data)]) == 1
