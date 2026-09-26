"""A tiny made-up city for the tests.

Every vehicle drives along its own straight street to the east, stops are about 310 m
apart. The real delays are set by the test, so the answers are known exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DAY = 1_767_657_600.0  # 2026-01-06 00:00:00 UTC
LAT = 55.75
LON0 = 37.60
LON_STEP = 0.005  # about 313 m at this latitude


def make_plan(tr_id=1, n_stops=30, start=DAY + 6 * 3600, every_s=60, layovers=(), row=0):
    """Planned stops of one vehicle. ``layovers`` is a list of ``(after_stop, pause_s)``."""
    times = []
    t = start
    pauses = dict(layovers)
    for k in range(n_stops):
        times.append(t)
        t += every_s + pauses.get(k, 0)
    return pd.DataFrame(
        {
            "stop_id": tr_id * 1000 + np.arange(n_stops),
            "tr_id": tr_id,
            "t_plan": np.array(times, dtype=float),
            "lat": LAT + row * 0.01,
            "lon": LON0 + LON_STEP * np.arange(n_stops),
            "address": [f"street {tr_id}, {k}" for k in range(n_stops)],
        }
    )


def make_track(plan, delays, dwell_s=20, step_s=10):
    """Clean telemetry of a bus that reaches stop ``k`` at ``t_plan[k] + delays[k]``."""
    plan = plan.sort_values("t_plan")
    arrive = plan["t_plan"].to_numpy() + np.asarray(delays, dtype=float)
    leave = arrive + dwell_s
    if np.any(leave[:-1] >= arrive[1:]):
        raise ValueError("delays make the bus leave a stop after it reaches the next one")
    lon = plan["lon"].to_numpy()
    lat = plan["lat"].to_numpy()

    t = np.arange(arrive[0] - 60, arrive[-1] + 60, step_s, dtype=float)
    fix_lon = np.interp(t, np.ravel(np.column_stack([arrive, leave])), np.repeat(lon, 2))
    fix_lat = np.interp(t, np.ravel(np.column_stack([arrive, leave])), np.repeat(lat, 2))
    speed = np.zeros_like(t)
    for k in range(len(arrive) - 1):
        moving = (t > leave[k]) & (t < arrive[k + 1])
        metres = 313.0 * abs(lon[k + 1] - lon[k]) / LON_STEP
        speed[moving] = metres / (arrive[k + 1] - leave[k]) * 3.6
    return pd.DataFrame(
        {
            "tr_id": int(plan["tr_id"].iloc[0]),
            "t": t,
            "ok": True,
            "lat": fix_lat,
            "lon": fix_lon,
            "speed": speed,
        }
    )


def stamp(seconds) -> pd.Series:
    """Seconds since the epoch -> strings in the organisers' format."""
    return pd.Series(pd.to_datetime(np.asarray(seconds), unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")


def write_dataset(root: Path, seed: int = 0) -> Path:
    """Write a fake dataset in the organisers' layout and return its root.

    Two real vehicles and one synthetic, the same vehicles in every part, points every five
    minutes, split between the parts in runs, like the real data.
    """
    rng = np.random.default_rng(seed)
    vehicles = {101: 0, 102: 1, 9_000_001: 2}
    plans, tracks, facts = [], [], []
    for tr_id, row in vehicles.items():
        plan = make_plan(tr_id, n_stops=150, every_s=60, layovers=[(49, 600), (99, 600)], row=row)
        # delays grow during a trip and drop at the terminal, some go past 2 minutes
        drift = np.cumsum(rng.normal(4.0, 8.0, len(plan)))
        delays = np.clip(drift - np.repeat([0.0, 150.0, 300.0], 50), -60, 300)
        plans.append(plan)
        tracks.append(make_track(plan, delays))
        facts.append(plan["t_plan"].to_numpy() + delays)

    plan = pd.concat(plans, ignore_index=True)
    fact = np.concatenate(facts)
    telemetry = pd.concat(tracks, ignore_index=True)

    points = []
    for tr_id in vehicles:
        own = plan[plan["tr_id"] == tr_id].reset_index(drop=True)
        own_fact = fact[plan["tr_id"].to_numpy() == tr_id]
        for T in np.arange(own["t_plan"].iloc[0], own["t_plan"].iloc[-1] - 900, 300.0):
            window = own.index[(own["t_plan"] > T + 600) & (own["t_plan"] <= T + 900)]
            passed = own.index[own["t_plan"] <= T]
            if len(window) == 0 or len(passed) == 0:
                continue
            target, last = window[0], passed[-1]
            points.append(
                {
                    "sample_id": f"{tr_id}_{int(T)}",
                    "tr_id": tr_id,
                    "T": T,
                    "target_stop_id": own.loc[target, "stop_id"],
                    "target_time_begin": own.loc[target, "t_plan"],
                    "cur_dev_s": own_fact[last] - own.loc[last, "t_plan"],
                    "target_delay_s": own_fact[target] - own.loc[target, "t_plan"],
                }
            )
    points = pd.DataFrame(points)
    run = (np.arange(len(points)) // 4) % 3
    parts = {"train": points[run == 0], "test": points[run == 1], "validate": points[run == 2]}
    parts["test"] = parts["test"][parts["test"]["tr_id"] < 9_000_000]
    parts["validate"] = parts["validate"][parts["validate"]["tr_id"] < 9_000_000]

    traffic = pd.DataFrame(
        {
            "packet_id": np.arange(len(telemetry)),
            "tr_id": telemetry["tr_id"],
            "unit_id": telemetry["tr_id"] + 500,
            "event_time": stamp(telemetry["t"]),
            "device_event_id": 0,
            "location_valid": True,
            "gps_time": stamp(telemetry["t"]),
            "lon": telemetry["lon"],
            "lat": telemetry["lat"],
            "alt": 150.0,
            "speed": telemetry["speed"],
            "heading": 90.0,
            "receive_time": stamp(telemetry["t"] + 1),
            "is_hist_data": False,
        }
    )
    schedule = pd.DataFrame(
        {
            "tt_action_item_id": plan["stop_id"],
            "time_begin": stamp(plan["t_plan"]),
            "time_fact_begin": stamp(fact),
            "order_date": "2026-01-06",
            "manual_fill": False,
            "tr_id": plan["tr_id"],
            "geom": "POINT (" + plan["lon"].astype(str) + " " + plan["lat"].astype(str) + ")",
            "building_address": plan["address"],
        }
    )

    for name in ("train", "test", "validate", "labels"):
        (root / name).mkdir(parents=True, exist_ok=True)
    real = traffic["tr_id"] < 9_000_000
    traffic.to_csv(root / "train/traffic.csv", index=False)
    traffic[real].to_csv(root / "test/traffic.csv", index=False)
    traffic[real].to_csv(root / "validate/traffic.csv", index=False)
    real_plan = schedule["tr_id"] < 9_000_000
    schedule.to_csv(root / "train/schedule.csv", index=False)
    schedule[real_plan].to_csv(root / "test/schedule.csv", index=False)
    schedule[real_plan].drop(columns="time_fact_begin").to_csv(
        root / "validate/schedule_plan.csv", index=False
    )

    for name, frame in parts.items():
        frame = frame.assign(
            T=stamp(frame["T"]).values, target_time_begin=stamp(frame["target_time_begin"]).values
        )
        if name == "validate":
            frame.drop(columns="target_delay_s").to_csv(root / "validate/points.csv", index=False)
        else:
            frame.assign(target_class="ontime").to_csv(
                root / f"labels/labels_{name}.csv", index=False
            )
    return root
