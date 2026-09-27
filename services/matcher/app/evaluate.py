"""Evaluate the live matcher detector against organizer CSVs without infrastructure."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from app.config import load_matcher_settings
from app.dataset import Tick, Visit
from app.matching import StopDetector
from common.config import load_dataset_config

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = re.compile(
    r"\s*POINT\s*\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    vehicle_id: str
    at: datetime
    reference_s: float | None


def rows(path: Path, required: set[str]):
    """Reject malformed CSV rows instead of silently improving coverage."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = required - set(headers)
        if missing or len(headers) != len(set(headers)):
            raise ValueError(f"{path.name}: missing {sorted(missing)} or duplicate headers")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}:{reader.line_num}: inconsistent column count")
            yield reader.line_num, row


def timestamp(value: str, zone: ZoneInfo) -> datetime:
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        first = parsed.replace(tzinfo=zone, fold=0)
        second = parsed.replace(tzinfo=zone, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ValueError("ambiguous local timestamp")
        parsed = first
    return parsed.astimezone(UTC)


def identifier(value: str) -> str:
    """Keep integer IDs exact, including values exported with a .0 suffix."""
    from decimal import Decimal, InvalidOperation

    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("invalid identifier") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("invalid identifier")
    return str(int(number))


def optional_number(value: str) -> float | None:
    if value.strip().lower() in {"", "nan", "null", "none"}:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def read_plan(path: Path, zone: ZoneInfo) -> list[Visit]:
    visits = []
    seen = set()
    for line, row in rows(path, {"tr_id", "tt_action_item_id", "time_begin", "geom"}):
        try:
            match = GEOMETRY.fullmatch(row["geom"])
            if match is None:
                raise ValueError("geom must be WKT POINT (lon lat)")
            lon, lat = float(match[1]), float(match[2])
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError("stop coordinates out of range")
            visit_id = identifier(row["tt_action_item_id"])
            if visit_id in seen:
                raise ValueError("duplicate stop id")
            seen.add(visit_id)
            visits.append(
                Visit(
                    identifier(row["tr_id"]),
                    visit_id,
                    timestamp(row["time_begin"], zone),
                    lon,
                    lat,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}: {exc}") from exc
    if not visits:
        raise ValueError(f"{path.name}: empty plan")
    return visits


def read_samples(path: Path, zone: ZoneInfo) -> list[Sample]:
    samples = []
    seen = set()
    for line, row in rows(path, {"sample_id", "tr_id", "T", "cur_dev_s"}):
        try:
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen:
                raise ValueError("blank or duplicate sample_id")
            seen.add(sample_id)
            samples.append(
                Sample(
                    sample_id,
                    identifier(row["tr_id"]),
                    timestamp(row["T"], zone),
                    optional_number(row["cur_dev_s"]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}: {exc}") from exc
    if not samples:
        raise ValueError(f"{path.name}: no scoring points")
    return samples


def read_traffic(path: Path, zone: ZoneInfo) -> tuple[dict[str, list[Tick]], int]:
    """Sort each vehicle by event time; at equal times prefer usable GPS."""
    vehicles: dict[str, list[Tick]] = defaultdict(list)
    count = 0
    required = {"tr_id", "event_time", "location_valid", "lon", "lat", "speed"}
    for line, row in rows(path, required):
        count += 1
        try:
            if not row["tr_id"].strip():
                continue
            flag = row["location_valid"].strip().lower()
            if flag not in {"0", "1", "false", "true"}:
                raise ValueError("invalid location_valid")
            lon = optional_number(row["lon"])
            lat = optional_number(row["lat"])
            if flag in {"0", "false"} or lon is None or lat is None:
                lon = lat = None
            elif not -180 <= lon <= 180 or not -90 <= lat <= 90:
                lon = lat = None
            speed = optional_number(row["speed"])
            if speed is not None and speed < 0:
                speed = None
            vehicle = identifier(row["tr_id"])
            vehicles[vehicle].append(
                Tick(vehicle, timestamp(row["event_time"], zone), lon, lat, speed)
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}: {exc}") from exc
    if not count:
        raise ValueError(f"{path.name}: no telemetry rows")
    for vehicle, ticks in vehicles.items():
        by_time = {}
        for tick in sorted(
            ticks,
            key=lambda item: (
                item.valid,
                item.speed is not None,
                item.lon or 0,
                item.lat or 0,
                item.speed if item.speed is not None else -1,
            ),
        ):
            by_time[tick.at] = tick
        vehicles[vehicle] = sorted(by_time.values(), key=lambda item: item.at)
    return dict(vehicles), count


def read_facts(path: Path, zone: ZoneInfo) -> dict[str, datetime] | None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if "time_fact_begin" not in (csv.DictReader(stream).fieldnames or []):
            return None
    facts = {}
    for line, row in rows(path, {"tt_action_item_id", "time_fact_begin"}):
        try:
            value = row["time_fact_begin"].strip()
            if value.lower() not in {"", "nan", "null", "none"}:
                facts[identifier(row["tt_action_item_id"])] = timestamp(value, zone)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}: invalid fact: {exc}") from exc
    return facts


def metrics(errors: list[float]) -> dict:
    absolute = sorted(abs(error) for error in errors)
    p95 = None
    if absolute:
        position = (len(absolute) - 1) * 0.95
        left = int(position)
        right = min(left + 1, len(absolute) - 1)
        p95 = absolute[left] + (absolute[right] - absolute[left]) * (position - left)
    return {
        "scored": len(errors),
        "mae_s": fmean(absolute) if absolute else None,
        "bias_s": fmean(errors) if errors else None,
        "p95_abs_error_s": p95,
    }


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(
    data_dir: Path,
    split: str,
    system_config: Path,
    assumptions: Path,
    output: Path,
) -> dict:
    """Replay without facts or labels in the detector, then compare its output."""
    if split not in {"train", "test"}:
        raise ValueError("quality evaluation supports train and test only")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    paths = {
        "schedule": data_dir / split / "schedule.csv",
        "traffic": data_dir / split / "traffic.csv",
        "labels": data_dir / "labels" / f"labels_{split}.csv",
        "system_config": system_config,
        "assumptions": assumptions,
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"required file not found: {path}")
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()}
    dataset = load_dataset_config(system_config)
    settings = load_matcher_settings(assumptions)
    zone = dataset.zone
    visits = read_plan(paths["schedule"], zone)
    samples = read_samples(paths["labels"], zone)
    traffic, traffic_rows = read_traffic(paths["traffic"], zone)
    detector = StopDetector(visits, settings)
    by_vehicle: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
    for position, sample in enumerate(samples):
        by_vehicle[sample.vehicle_id].append((position, sample))
    comparisons: list[dict | None] = [None] * len(samples)
    bounds = {}
    for vehicle in sorted(set(traffic) | set(by_vehicle)):
        ticks = traffic.get(vehicle, [])
        if ticks:
            bounds[vehicle] = (ticks[0].at, ticks[-1].at)
        ordered_samples = sorted(by_vehicle.get(vehicle, []), key=lambda pair: pair[1].at)
        index = 0
        latest = None

        def feed_one(item: Tick) -> None:
            nonlocal latest
            for event in detector.feed(item):
                if latest is None or event.arrival >= latest.arrival:
                    latest = event

        for position, sample in ordered_samples:
            while index < len(ticks) and ticks[index].at <= sample.at:
                feed_one(ticks[index])
                index += 1
            estimate = latest.delay_sec if latest is not None else None
            comparisons[position] = {
                "sample_id": sample.sample_id,
                "tr_id": sample.vehicle_id,
                "T": sample.at.isoformat(),
                "estimate_s": estimate,
                "reference_s": sample.reference_s,
                "last_stop_id": latest.visit_id if latest is not None else None,
                "last_time_fact": latest.arrival.isoformat() if latest is not None else None,
                "error_s": estimate - sample.reference_s
                if estimate is not None and sample.reference_s is not None
                else None,
            }
        while index < len(ticks):
            feed_one(ticks[index])
            index += 1
    events = detector.events
    facts = read_facts(paths["schedule"], zone)
    scored = [row["error_s"] for row in comparisons if row["error_s"] is not None]
    covered = sum(row["estimate_s"] is not None for row in comparisons)
    arrivals = {"detected": len(events), "reference_available": facts is not None}
    if facts is not None:
        planned = {visit.visit_id: visit for visit in visits}
        expected = {
            stop_id: fact
            for stop_id, fact in facts.items()
            if stop_id in planned
            and planned[stop_id].vehicle_id in bounds
            and bounds[planned[stop_id].vehicle_id][0]
            <= fact
            <= bounds[planned[stop_id].vehicle_id][1]
        }
        found = {event.visit_id: event for event in events}
        matched = expected.keys() & found.keys()
        arrivals.update(
            {
                "reference_in_observed_window": len(expected),
                "matched_by_stop_id": len(matched),
                "recall_by_stop_id": len(matched) / len(expected) if expected else None,
                "time_error": metrics(
                    [
                        (found[stop_id].arrival - expected[stop_id]).total_seconds()
                        for stop_id in matched
                    ]
                ),
            }
        )
    report = {
        "schema_version": 1,
        "split": split,
        "quality_gate": "not_configured",
        "source_timezone": dataset.source_timezone,
        "settings": asdict(settings),
        "implementation_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("matching.py", "evaluate.py", "dataset.py", "config.py")
        },
        "inputs": {
            name: {"sha256": digest(path), "bytes": before[name][0]} for name, path in paths.items()
        },
        "points": {
            "total": len(samples),
            "covered": covered,
            "coverage": covered / len(samples),
            "missing_reference": sum(row["reference_s"] is None for row in comparisons),
            **metrics(scored),
        },
        "arrivals": arrivals,
        "traffic_rows": traffic_rows,
        "deduplicated_ticks": sum(map(len, traffic.values())),
        "rejected_late_ticks": detector.rejected_late_ticks,
    }
    for name, path in paths.items():
        if before[name] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"input changed during replay: {path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="matcher-report-", dir=output.parent) as scratch:
        staged = Path(scratch) / "report"
        staged.mkdir()
        (staged / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staged / "points.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        with (staged / "events.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = [
                "tr_id",
                "stop_id",
                "time_plan",
                "time_fact",
                "available_at",
                "departure",
                "recovered",
                "delay_s",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {
                        "tr_id": event.vehicle_id,
                        "stop_id": event.visit_id,
                        "time_plan": event.planned_at.isoformat(),
                        "time_fact": event.arrival.isoformat(),
                        "available_at": event.available_at.isoformat(),
                        "departure": event.departure.isoformat() if event.departure else None,
                        "recovered": event.recovered,
                        "delay_s": event.delay_sec,
                    }
                )
        staged.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценить текущую задержку по событиям матчера.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/system.yaml")
    parser.add_argument("--assumptions", type=Path, default=ROOT / "config/assumptions.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / "data/reports/matcher" / (
        f"{args.split}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    try:
        report = evaluate(args.data_dir, args.split, args.config, args.assumptions, output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка оценки: {exc}\n")
    points = report["points"]
    print(f"Отчёт: {output.resolve()}")
    print(f"Точек: {points['total']}; покрыто: {points['covered']}; MAE: {points['mae_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
