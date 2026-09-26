import numpy as np
import pandas as pd
import pytest

from busdelay.data import clean_telemetry, is_synthetic, to_seconds
from busdelay.geo import haversine_m, parse_points


def test_haversine_one_degree_of_latitude():
    assert haversine_m(55.0, 37.0, 56.0, 37.0) == pytest.approx(111_195, rel=1e-3)
    assert haversine_m(55.75, 37.6, 55.75, 37.6) == 0


def test_parse_points():
    coords = parse_points(pd.Series(["POINT (37.43070705 55.8040083)", "POINT(37.5 55.7)"]))
    assert coords["lon"].tolist() == [37.43070705, 37.5]
    assert coords["lat"].tolist() == [55.8040083, 55.7]
    with pytest.raises(ValueError):
        parse_points(pd.Series(["LINESTRING (1 2, 3 4)"]))


def test_to_seconds_matches_sample_id_suffix():
    # sample_id 122048_1767665400 has T = 2026-01-06 02:10:00
    assert to_seconds(["2026-01-06 02:10:00"])[0] == 1767665400
    both = to_seconds(["2026-01-06 02:10:00.500000", "2026-01-06 02:10:01"])
    assert both.tolist() == [1767665400.5, 1767665401.0]


def test_clean_telemetry():
    raw = pd.DataFrame(
        {
            "tr_id": [1, 1, 1, 1, 2],
            "event_time": [
                "2026-01-06 10:00:20",
                "2026-01-06 10:00:10",
                "2026-01-06 10:00:10",
                "2026-01-06 10:00:30",
                "2026-01-06 10:00:00",
            ],
            "location_valid": [True, True, True, False, True],
            "lat": [55.7, 55.7, 55.7, 0.0, 10.0],
            "lon": [37.6, 37.6, 37.6, 0.0, 37.6],
            "speed": [20.0, 300.0, 300.0, 0.0, 5.0],
        }
    )
    clean = clean_telemetry(raw)
    assert clean["tr_id"].tolist() == [1, 1, 1, 2]  # duplicate dropped, sorted
    assert np.diff(clean["t"].to_numpy()[:3]).min() > 0
    assert clean["ok"].tolist() == [True, True, False, False]  # invalid flag, outside bbox
    assert np.isnan(clean["speed"].iloc[0])  # 300 km/h is not a bus
    assert clean["lat"].isna().tolist() == [False, False, True, True]


def test_synthetic_vehicles():
    assert is_synthetic([131672, 9_000_000]).tolist() == [False, True]
