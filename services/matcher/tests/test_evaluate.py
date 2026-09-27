import csv
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.evaluate import evaluate, read_plan

START = datetime(2026, 1, 6, 12, tzinfo=UTC)


def at(seconds):
    return (START + timedelta(seconds=seconds)).replace(tzinfo=None).isoformat()


def write_csv(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    schedule = root / "train/schedule.csv"
    traffic = root / "train/traffic.csv"
    labels = root / "labels/labels_train.csv"
    write_csv(
        schedule,
        ["tt_action_item_id", "tr_id", "time_begin", "geom", "time_fact_begin"],
        [
            {
                "tt_action_item_id": "9007199254740993",
                "tr_id": "7",
                "time_begin": at(0),
                "geom": "POINT (37 55)",
                "time_fact_begin": at(0),
            }
        ],
    )
    write_csv(
        traffic,
        ["tr_id", "event_time", "location_valid", "lon", "lat", "speed"],
        [
            {
                "tr_id": "7",
                "event_time": at(20),
                "location_valid": "1",
                "lon": "37.01",
                "lat": "55",
                "speed": "20",
            },
            {
                "tr_id": "7",
                "event_time": at(0),
                "location_valid": "1",
                "lon": "37",
                "lat": "55",
                "speed": "20",
            },
        ],
    )
    write_csv(
        labels,
        ["sample_id", "tr_id", "T", "cur_dev_s", "target_delay_s"],
        [
            {
                "sample_id": "before",
                "tr_id": "7",
                "T": at(10),
                "cur_dev_s": "0",
                "target_delay_s": "999",
            },
            {
                "sample_id": "after",
                "tr_id": "7",
                "T": at(20),
                "cur_dev_s": "0",
                "target_delay_s": "-999",
            },
        ],
    )
    system = tmp_path / "system.yaml"
    system.write_text("dataset:\n  source_timezone: UTC\n", encoding="utf-8")
    assumptions = tmp_path / "assumptions.yaml"
    assumptions.write_text(
        "matcher:\n"
        "  stop_radius_m: 50\n"
        "  stopped_speed_kmh: 5\n"
        "  visit_time_tolerance_sec: 300\n"
        "  stale_after_sec: 300\n",
        encoding="utf-8",
    )
    return root, system, assumptions


def run(dataset, output):
    root, system, assumptions = dataset
    return evaluate(root, "train", system, assumptions, output)


def test_recovered_arrival_is_available_only_after_exit(dataset, tmp_path):
    output = tmp_path / "report"
    report = run(dataset, output)
    assert report["points"]["total"] == 2
    assert report["points"]["covered"] == 1
    assert report["points"]["coverage"] == 0.5
    assert report["points"]["mae_s"] == 0
    assert report["arrivals"]["detected"] == 1
    assert report["arrivals"]["time_error"]["mae_s"] == 0
    rows = read_csv(output / "points.csv")
    assert rows[0]["estimate_s"] == ""
    assert rows[1]["estimate_s"] == "0.0"
    event = read_csv(output / "events.csv")[0]
    assert event["time_fact"] == START.isoformat()
    assert event["available_at"] == (START + timedelta(seconds=20)).isoformat()
    assert event["recovered"] == "True"
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_facts_and_targets_do_not_change_detector_output(dataset, tmp_path):
    root, system, assumptions = dataset
    first = tmp_path / "first"
    evaluate(root, "train", system, assumptions, first)
    schedule = read_csv(root / "train/schedule.csv")
    schedule[0]["time_fact_begin"] = at(100)
    write_csv(root / "train/schedule.csv", list(schedule[0]), schedule)
    labels = read_csv(root / "labels/labels_train.csv")
    for row in labels:
        row["cur_dev_s"] = "12345"
        row["target_delay_s"] = "67890"
    write_csv(root / "labels/labels_train.csv", list(labels[0]), labels)
    second = tmp_path / "second"
    evaluate(root, "train", system, assumptions, second)
    assert (first / "events.csv").read_bytes() == (second / "events.csv").read_bytes()
    assert [row["estimate_s"] for row in read_csv(first / "points.csv")] == [
        row["estimate_s"] for row in read_csv(second / "points.csv")
    ]


def test_cli_runs_without_database_or_redis(dataset, tmp_path, monkeypatch):
    root, system, assumptions = dataset
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.evaluate",
            "--data-dir",
            str(root),
            "--split",
            "train",
            "--config",
            str(system),
            "--assumptions",
            str(assumptions),
            "--output",
            str(tmp_path / "report"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr


def test_bad_geometry_and_existing_report_are_rejected(dataset, tmp_path):
    root, system, assumptions = dataset
    schedule = read_csv(root / "train/schedule.csv")
    schedule[0]["geom"] = "invalid"
    write_csv(root / "train/schedule.csv", list(schedule[0]), schedule)
    with pytest.raises(ValueError, match="schedule.csv:2:.*WKT"):
        read_plan(root / "train/schedule.csv", UTC)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        evaluate(root, "train", system, assumptions, output)
    assert list(output.iterdir()) == []


def test_validate_is_not_used_as_quality_data(dataset, tmp_path):
    root, system, assumptions = dataset
    with pytest.raises(ValueError, match="train and test only"):
        evaluate(root, "validate", system, assumptions, tmp_path / "report")
