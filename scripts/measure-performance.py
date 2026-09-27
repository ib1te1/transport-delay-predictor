"""Measure the whole compose stack on a replay of the demo period.

Runs on the host from the repository root, with Docker and Python 3.13 and nothing beyond the
standard library:

    python scripts/measure-performance.py

1. Cold start: ``docker compose down -v``, then ``up -d --wait``; every service is timed until
   its HTTP endpoint answers. Images are built beforehand, the build is not timed.
2. Replay of ``replay.period`` from config/system.yaml at ``replay.speedup`` (or
   ``--speedup``). Every few seconds: CPU and memory of the containers; every second: how
   long telemetry waits in the stream for matcher.
3. At 3/4 of the replay predictor is stopped for ``--outage-sec`` and started again: the api
   has to fall back to the baseline and return to the model by itself.
4. Figures from predictor's ``GET /metrics``, the ``predictions`` and ``telemetry`` streams
   and the replay log go between the markers in docs/performance.md, raw samples to
   data/perf/.

Step 1 deletes the Postgres volume: whatever the running demo holds is lost.
"""

import argparse
import csv
import http.client
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "performance.md"
OUT = ROOT / "data" / "perf"
BEGIN = "<!-- measure-performance:begin -->"
END = "<!-- measure-performance:end -->"

# What each service answers once it is up, on the ports docker-compose.yml publishes.
READY = {
    "predictor": "http://127.0.0.1:8003/health",
    "api": "http://127.0.0.1:8000/health",
    "ingest": "http://127.0.0.1:8001/health",
    "matcher": "http://127.0.0.1:8002/ready",
    "web": "http://127.0.0.1:5173/",
}
API = "http://127.0.0.1:8000"
PREDICTOR = "http://127.0.0.1:8003"
SAMPLE_SEC = 5.0
OUTAGE_AT = 0.75
PAGE = 10000

SAMPLER_NAME = "measure-performance-sampler"
# Runs in a container of the api image. Once a second: how long the oldest telemetry entry
# matcher has not taken yet has been waiting in the stream. The cursor and the stream are
# read a few milliseconds apart, on the Redis clock.
SAMPLER = """
import json, os, time
import psycopg, redis
bus = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
db = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
while True:
    row = db.execute("SELECT stream_id FROM matcher_cursor").fetchone()
    waiting = bus.xrange("telemetry", "(" + row[0], "+", count=1) if row else []
    sec, usec = bus.time()
    now = sec * 1000 + usec // 1000
    age = now - int(waiting[0][0].split("-")[0]) if waiting else 0
    print(json.dumps({"ms": now, "age_ms": age if row else None}), flush=True)
    time.sleep(1)
"""


def run(args: list[str], timeout: float = 600, check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        sys.exit(
            f"{' '.join(args)} exited with {result.returncode}\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def compose(*args: str, **kwargs) -> str:
    return run(["docker", "compose", *args], **kwargs)


def redis(*args: str):
    out = compose("exec", "-T", "redis", "redis-cli", "--json", *args)
    return json.loads(out) if out.strip() else None


def redis_ms() -> int:
    """Redis server clock: stream ids carry the same one, so the host clock never mixes in."""
    sec, usec = redis("TIME")
    return int(sec) * 1000 + int(usec) // 1000


def matcher_cursor() -> str | None:
    query = "SELECT stream_id FROM matcher_cursor"
    script = 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -c "$0"'
    out = compose("exec", "-T", "postgres", "sh", "-c", script, query).strip()
    return out or None


def get_json(url: str, timeout: float = 5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (OSError, http.client.HTTPException, ValueError):
        return None


def answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def id_ms(entry_id: str) -> int:
    return int(entry_id.split("-")[0])


def seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def pct(values: list[float], q: float) -> float:
    """Percentile with linear interpolation, as numpy's default."""
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def read_stream(name: str) -> list[tuple[int, dict]]:
    out, lower = [], "-"
    while True:
        page = redis("XRANGE", name, lower, "+", "COUNT", str(PAGE)) or []
        for entry_id, fields in page:
            data = dict(zip(fields[::2], fields[1::2], strict=True))["data"]
            out.append((id_ms(entry_id), json.loads(data)))
        if len(page) < PAGE:
            return out
        lower = "(" + page[-1][0]


def compose_rows(*args: str) -> list[dict]:
    """``--format json`` output: one array in older compose, one object a line in newer."""
    out = compose(*args, "--format", "json").strip()
    if out.startswith("["):
        return json.loads(out)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def task_seconds(container: str) -> float:
    out = run(["docker", "inspect", "-f", "{{.State.StartedAt}} {{.State.FinishedAt}}", container])
    started, finished = (re.sub(r"(\.\d{6})\d*", r"\1", s) for s in out.split())
    return seconds(finished) - seconds(started)


def cold_start() -> dict:
    compose("--profile", "*", "down", "-v", "--remove-orphans")
    log = (OUT / "compose-up.log").open("w", encoding="utf-8")
    start = time.monotonic()
    up = subprocess.Popen(
        ["docker", "compose", "up", "-d", "--wait", "--wait-timeout", "300"],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    ready: dict[str, float] = {}
    waited = None
    while waited is None or len(ready) < len(READY):
        now = time.monotonic() - start
        if waited is None and up.poll() is not None:
            waited = now
            if up.returncode != 0:
                sys.exit(f"docker compose up failed, see {log.name}")
        for name, url in READY.items():
            if name not in ready and answers(url):
                ready[name] = round(now, 1)
        if now > 330:
            sys.exit(f"services not up after {now:.0f} s: {sorted(set(READY) - set(ready))}")
        time.sleep(0.2)
    log.close()
    names = {row["Service"]: row["Name"] for row in compose_rows("ps", "-a")}
    tasks = {name: round(task_seconds(names[name]), 1) for name in ("migrate", "seed")}
    return {"ready_s": ready, "up_wait_s": round(waited, 1), "tasks_s": tasks}


def read_config() -> dict:
    code = (
        "import json, yaml; "
        "print(json.dumps(yaml.safe_load(open('/config/system.yaml')), default=str))"
    )
    return json.loads(compose("exec", "-T", "api", "python", "-c", code))


def host_info() -> dict:
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"]))
    cpu = compose("exec", "-T", "api", "sh", "-c", "grep -m1 'model name' /proc/cpuinfo")
    return {
        "cpu": cpu.split(":", 1)[-1].strip(),
        "ncpu": info["NCPU"],
        "mem_gb": round(info["MemTotal"] / 2**30, 1),
        "docker": f"{info['OperatingSystem']} {info['ServerVersion']}",
    }


def dataset_span(config: dict) -> tuple[float, float]:
    """First and last telemetry time of the replayed period, within start_at and end_at."""
    zone_name = config["dataset"]["source_timezone"]
    zone = UTC if zone_name == "UTC" else ZoneInfo(zone_name)
    replay = config["replay"]
    path = ROOT / "data" / "dataset" / replay["period"] / "traffic.csv"
    with path.open(encoding="utf-8", newline="") as f:
        times = [row["event_time"] for row in csv.DictReader(f)]
    bounds = [min(times), max(times)]
    if replay.get("start_at"):
        bounds[0] = max(bounds[0], replay["start_at"])
    if replay.get("end_at"):
        bounds[1] = min(bounds[1], replay["end_at"])
    lo, hi = (datetime.fromisoformat(b).replace(tzinfo=zone).timestamp() for b in bounds)
    return lo, hi


def replay_config(speedup: float | None) -> list[str]:
    """Extra ``compose run`` arguments that give the replay another speedup."""
    if speedup is None:
        return []
    conf = OUT / "config"
    shutil.copytree(ROOT / "config", conf, dirs_exist_ok=True)
    text = (conf / "system.yaml").read_text(encoding="utf-8")
    head, sep, tail = text.partition("\nreplay:\n")
    tail, n = re.subn(
        r"^([ \t]+speedup:)[ \t]*\S+", rf"\g<1> {speedup:g}", tail, count=1, flags=re.M
    )
    if not sep or n != 1:
        sys.exit("config/system.yaml has no replay.speedup to override")
    (conf / "system.yaml").write_text(head + sep + tail, encoding="utf-8")
    return ["-v", f"{conf}:/perf-config:ro", "-e", "CONFIG_PATH=/perf-config/system.yaml"]


def mib(value: str) -> float:
    units = {"B": 2**-20, "KiB": 2**-10, "kB": 2**-10, "MiB": 1, "MB": 1, "GiB": 1024, "GB": 1024}
    number, unit = re.fullmatch(r"([\d.]+)\s*(\w+)", value.strip()).groups()
    return float(number) * units[unit]


def container_stats(project: str) -> dict[str, tuple[float, float]]:
    """Service -> (CPU % of one core, memory MiB), one-off runs such as replay included."""
    label = f"label=com.docker.compose.project={project}"
    service = {}
    for line in run(["docker", "ps", "--filter", label, "--format", "{{json .}}"]).splitlines():
        row = json.loads(line)
        if row["Names"] == SAMPLER_NAME:
            continue
        labels = dict(item.split("=", 1) for item in row["Labels"].split(",") if "=" in item)
        service[row["Names"]] = labels.get("com.docker.compose.service", row["Names"])
    if not service:
        return {}
    out = run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *service])
    stats = {}
    for line in out.splitlines():
        row = json.loads(line)
        cpu = float(row["CPUPerc"].rstrip("%"))
        stats[service[row["Name"]]] = (cpu, mib(row["MemUsage"].split("/")[0]))
    return stats


def start_sampler() -> None:
    """A container of its own, so its CPU does not land on the service it watches."""
    compose(
        "run", "-d", "--rm", "--no-deps", "-T", "--name", SAMPLER_NAME, "api",
        "python", "-c", SAMPLER,
    )  # fmt: skip


def stop_sampler() -> list[dict]:
    out = run(["docker", "logs", SAMPLER_NAME])
    run(["docker", "rm", "-f", SAMPLER_NAME])
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def sample(start: float, project: str) -> dict:
    state = get_json(f"{API}/api/state") or {}
    return {
        "t": round(time.monotonic() - start, 1),
        "clock": state.get("clock"),
        "containers": container_stats(project),
    }


def outage(seconds_off: float, clock: str) -> dict:
    before = get_json(f"{PREDICTOR}/metrics")
    stop_ms = redis_ms()
    compose("stop", "predictor")
    time.sleep(seconds_off)
    start_ms = redis_ms()
    compose("start", "predictor")
    deadline = time.monotonic() + 120
    while not answers(READY["predictor"]) and time.monotonic() < deadline:
        time.sleep(0.2)
    return {
        "metrics_before": before,
        "stop_ms": stop_ms,
        "start_ms": start_ms,
        "ready_ms": redis_ms(),
        "seconds": seconds_off,
        "clock": clock,
    }


def replay(config: dict, speedup: float | None, outage_sec: float, project: str) -> dict:
    lo, hi = dataset_span(config)
    outage_at = lo + OUTAGE_AT * (hi - lo)
    log_path = OUT / "replay.log"
    command = ["docker", "compose", "--profile", "replay", "run", "--rm", "-T"]
    command += [*replay_config(speedup), "replay"]
    run(["docker", "rm", "-f", SAMPLER_NAME], check=False)
    start_sampler()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        samples, off = [], None
        while proc.poll() is None:
            began = time.monotonic()
            samples.append(sample(start, project))
            clock = samples[-1]["clock"]
            if off is None and outage_sec > 0 and clock and seconds(clock) >= outage_at:
                off = outage(outage_sec, clock)
            time.sleep(max(0.0, SAMPLE_SEC - (time.monotonic() - began)))
        wall = time.monotonic() - start
    if proc.returncode != 0:
        sys.exit(f"replay exited with {proc.returncode}, see {log_path}")
    # let matcher and the last tick finish
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        last = redis("XREVRANGE", "telemetry", "+", "-", "COUNT", "1")
        if last and matcher_cursor() == last[0][0]:
            break
        time.sleep(1)
    time.sleep(5)
    text = log_path.read_text(encoding="utf-8")
    lags = [float(x) for x in re.findall(r"replay lag_sec=([\d.]+)", text)]
    return {
        "samples": samples,
        "matcher": stop_sampler(),
        "outage": off,
        "wall_s": wall,
        "replay_lags_s": lags,
    }


def summary(values: list[float], digits: int = 0) -> dict:
    if not values:
        return {}
    out = {f"p{q}": round(pct(values, q), digits) for q in (50, 95, 99)}
    out["max"] = round(max(values), digits)
    return out


def analyse(run_: dict) -> dict:
    telemetry = read_stream("telemetry")
    predictions = read_stream("predictions")
    tel_ms = [ms for ms, _ in telemetry]
    clock, top = [], -math.inf
    for _, data in telemetry:
        top = max(top, seconds(data["event_time"]))
        clock.append(top)

    by_tick = defaultdict(list)
    for ms, row in predictions:
        by_tick[row["t"]].append((ms, row))
    ticks = []
    for t, rows in by_tick.items():
        first = bisect_left(clock, seconds(t))
        ticks.append(
            {
                "T": seconds(t),
                "n": len(rows),
                "published_ms": max(ms for ms, _ in rows),
                "clock_ms": tel_ms[first] if first < len(tel_ms) else None,
                "fallback": any(r["model_version"] == "fallback" for _, r in rows),
            }
        )
    ticks.sort(key=lambda tick: tick["T"])
    model = [t for t in ticks if not t["fallback"] and t["clock_ms"] is not None]
    lag = [t["published_ms"] - t["clock_ms"] for t in model]
    n = [t["n"] for t in model]
    if not model:
        sys.exit("no model predictions in the predictions stream")
    mean_n, mean_lag = sum(n) / len(n), sum(lag) / len(lag)
    slope = sum((a - mean_n) * (b - mean_lag) for a, b in zip(n, lag, strict=True)) / max(
        sum((a - mean_n) ** 2 for a in n), 1e-9
    )
    steps = [b["T"] - a["T"] for a, b in zip(ticks, ticks[1:], strict=False)]
    rows = [row for _, row in predictions]

    samples = run_["samples"]
    behind = [s["age_ms"] for s in run_["matcher"] if s["age_ms"] is not None]
    per_service = defaultdict(lambda: ([], []))
    for s in samples:
        for name, (cpu, mem) in s["containers"].items():
            per_service[name][0].append(cpu)
            per_service[name][1].append(mem)

    result = {
        "telemetry_rows": len(telemetry),
        "publish_span_s": (tel_ms[-1] - tel_ms[0]) / 1000,
        "dataset_span_s": clock[-1] - seconds(telemetry[0][1]["event_time"]),
        "replay_lags_s": run_["replay_lags_s"],
        "matcher_behind_ms": summary(behind),
        "ticks": len(ticks),
        "rows": len(rows),
        "vehicles": len({r["tr_id"] for r in rows}),
        "vehicles_per_tick": summary(n),
        "tick_step_s": summary(steps),
        "tick_lag_ms": summary(lag),
        "tick_lag_fit": {
            "base_ms": round(mean_lag - slope * mean_n),
            "per_vehicle_ms": round(slope, 1),
        },
        "degraded": dict(Counter(r["degraded_reason"] or "none" for r in rows)),
        "resources": {
            name: {"cpu": summary(cpu), "mem_max_mib": round(max(mem))}
            for name, (cpu, mem) in sorted(per_service.items())
        },
    }
    off = run_["outage"]
    if off:
        during = [t for t in ticks if t["fallback"]]
        back = [t for t in ticks if not t["fallback"] and t["published_ms"] > off["start_ms"]]

        def after(ms: int, since: int) -> float:
            return round((ms - since) / 1000, 1)

        result["outage"] = {
            "seconds": off["seconds"],
            "clock_at_stop": off["clock"],
            "fallback_ticks": len(during),
            "fallback_rows": sum(t["n"] for t in during),
            "first_fallback_s": after(during[0]["published_ms"], off["stop_ms"])
            if during
            else None,
            "ready_s": after(off["ready_ms"], off["start_ms"]),
            "model_back_s": after(back[0]["published_ms"], off["start_ms"]) if back else None,
        }
    return result


def fmt(d: dict, keys=("p50", "p95", "p99", "max")) -> str:
    return " | ".join(f"{d[k]:g}" if k in d else "—" for k in keys)


def render(meta: dict, cold: dict, figures: dict, metrics: dict, metrics_end: dict) -> str:
    host = meta["host"]
    predict = metrics["predict"]
    lines = [
        BEGIN,
        "",
        f"Прогон {meta['date']} UTC: период `{meta['period']}`, ускорение ×{meta['speedup']:g}, "
        f"модель `{metrics['model_version']}`. Стенд: {host['cpu']}, {host['docker']}: "
        f"{host['ncpu']} CPU и {host['mem_gb']} ГБ памяти для контейнеров.",
        "",
        "**Холодный старт.** `docker compose up -d --wait` после `down -v`, образы собраны "
        "заранее. Время — от команды до первого ответа сервиса по HTTP.",
        "",
        "| Сервис | Отвечает через, с |",
        "| --- | ---: |",
    ]
    for name, s in sorted(cold["ready_s"].items(), key=lambda kv: kv[1]):
        lines.append(f"| `{name}` | {s:g} |")
    model_ticks = figures["ticks"] - figures.get("outage", {}).get("fallback_ticks", 0)
    lines += [
        f"| `up --wait` вернул управление | {cold['up_wait_s']:g} |",
        "",
        f"Одноразовые задачи: `migrate` — {cold['tasks_s']['migrate']:g} с, "
        f"`seed` — {cold['tasks_s']['seed']:g} с.",
        "",
        "**Прогноз.** `/predict` — по `GET /metrics` у `predictor`, вызовов до отказа "
        f"(см. ниже): {predict['window']}. «Телеметрия → прогноз» — от записи в шине "
        "`telemetry`, которая сдвинула часы датасета до `T`, до публикации прогнозов тика "
        f"в шину `predictions`, тиков без отказа: {model_ticks}.",
        "",
        "| | p50 | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `/predict`, мс | {fmt(predict['latency_ms'])} |",
        f"| ТС в пачке | {fmt(predict['batch_size'])} |",
        f"| тело запроса, КБ | {fmt(predict['body_kb'])} |",
        f"| телеметрия → прогноз, мс | {fmt(figures['tick_lag_ms'])} |",
        f"| ТС в тике | {fmt(figures['vehicles_per_tick'])} |",
        "",
        f"Линейная оценка по тикам: телеметрия → прогноз ≈ {figures['tick_lag_fit']['base_ms']} мс "
        f"+ {figures['tick_lag_fit']['per_vehicle_ms']:g} мс на каждое ТС в тике.",
        "",
    ]
    lags = figures["replay_lags_s"]
    degraded = figures["degraded"]

    def slash(d: dict, keys: tuple[str, ...]) -> str:
        return fmt(d, keys).replace(" | ", " / ")

    lines += [
        "**Поток.**",
        "",
        "| Что | Значение |",
        "| --- | ---: |",
        f"| точек телеметрии | {figures['telemetry_rows']} |",
        f"| проиграно за, мин | {figures['publish_span_s'] / 60:.1f} "
        f"(×{figures['dataset_span_s'] / figures['publish_span_s']:.1f}) |",
        "| проигрыватель отстал от графика больше чем на 1 с | "
        + (f"{len(lags)} раз, до {max(lags):g} с |" if lags else "ни разу |"),
        "| телеметрия ждёт `matcher` в шине, мс: p50 / p95 / max | "
        f"{slash(figures['matcher_behind_ms'], ('p50', 'p95', 'max'))} |",
        f"| тиков / прогнозов / ТС | {figures['ticks']} / {figures['rows']} / "
        f"{figures['vehicles']} |",
        "| шаг тика по часам датасета, с: p50 / max | "
        f"{slash(figures['tick_step_s'], ('p50', 'max'))} |",
        f"| прогнозов без `degraded` | {degraded.get('none', 0)} |",
    ]
    for reason in sorted(set(degraded) - {"none"}):
        lines.append(f"| `degraded_reason = {reason}` | {degraded[reason]} |")
    lines.append("")
    off = figures.get("outage")
    if off:
        lines += [
            f"**Отказ `predictor`.** `docker compose stop predictor` в {off['clock_at_stop']} "
            f"по часам датасета, через {off['seconds']:g} с — `start`. Прогнозов по бейзлайну "
            f"с `predictor_unavailable`: {off['fallback_rows']} в {off['fallback_ticks']} "
            f"тиках, первый — через {off['first_fallback_s']} с после остановки. После "
            f"`start` сервис ответил через {off['ready_s']:g} с, первый прогноз модели — "
            f"через {off['model_back_s']} с. После рестарта `/predict` вызван "
            f"{metrics_end['predict']['calls']} раз, ошибок {metrics_end['predict']['failed']}.",
            "",
        ]
    lines += [
        f"**Ресурсы** — `docker stats` раз в {SAMPLE_SEC:g} с во время проигрывания. CPU — "
        "в процентах одного ядра.",
        "",
        "| Сервис | CPU p50, % | CPU max, % | Память max, МиБ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, r in figures["resources"].items():
        lines.append(
            f"| `{name}` | {r['cpu']['p50']:g} | {r['cpu']['max']:g} | {r['mem_max_mib']} |"
        )
    lines += ["", END]
    return "\n".join(lines)


def measure(speedup: float | None, outage_sec: float) -> dict:
    """The whole run; the raw figures are also saved to data/perf/."""
    OUT.mkdir(parents=True, exist_ok=True)
    print("building images (not timed)", flush=True)
    compose("build", timeout=1800)
    print("cold start", flush=True)
    cold = cold_start()
    config = read_config()
    project = json.loads(compose("config", "--format", "json"))["name"]
    meta = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "period": config["replay"]["period"],
        "speedup": speedup or config["replay"]["speedup"],
        "host": host_info(),
    }
    print(f"replay of {meta['period']} at x{meta['speedup']:g}", flush=True)
    run_ = replay(config, speedup, outage_sec, project)
    metrics_end = get_json(f"{PREDICTOR}/metrics")
    metrics = (run_["outage"] or {}).get("metrics_before") or metrics_end
    raw = {"meta": meta, "cold": cold, "figures": analyse(run_), "metrics": metrics}
    raw |= {"metrics_end": metrics_end, "run": run_, "config": config}
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (OUT / f"perf-{stamp}.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speedup", type=float, help="instead of replay.speedup")
    parser.add_argument("--outage-sec", type=float, default=20, help="0 skips the outage")
    parser.add_argument("--no-write", action="store_true", help="print, leave the doc alone")
    parser.add_argument("--render", type=Path, help="a saved data/perf/*.json instead of a run")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.render:
        raw = json.loads(args.render.read_text(encoding="utf-8"))
    else:
        raw = measure(args.speedup, args.outage_sec)
    block = render(raw["meta"], raw["cold"], raw["figures"], raw["metrics"], raw["metrics_end"])
    print(block)
    if args.no_write:
        return
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        sys.exit(f"{DOC} has no {BEGIN} ... {END} block to replace")
    head, rest = text.split(BEGIN, 1)
    DOC.write_text(head + block + rest.split(END, 1)[1], encoding="utf-8")
    print(f"written to {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
