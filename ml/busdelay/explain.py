"""Why the model expects a delay: contributions grouped into reasons for the dispatcher.

CatBoost gives SHAP values, a contribution in seconds for every feature of every row. The
features are grouped into a few reasons a dispatcher understands, and the reason with the
largest push towards being late is shown on the incident card.

With a residual model the prediction is ``baseline + trees`` (in a blend the baseline has a
weight below one). The baseline part is split too: ``a * cur_dev_s`` goes to the carried
delay, and the difference between the two lines of :class:`busdelay.baselines.SplitLinear`
goes to the layover.
"""

import numpy as np
import pandas as pd
from catboost import Pool

from .model import VEHICLE_COLUMN, DelayModel, make_matrix

# reason key -> (text for the dashboard, features)
REASONS = {
    "carried_delay": (
        "Накопленное опоздание",
        [
            "cur_dev_s",
            "gps_dev_last_s",
            "gps_dev_mean3_s",
            "gps_dev_trend_s",
            "gps_overdue_s",
            "gps_dev_now_s",
            "gps_minus_cur_s",
            "gps_dev_age_s",
        ],
    ),
    "layover": (
        "Отстой на конечной перед остановкой",
        ["layover_ahead_s", "new_trip_ahead", "layover_slack_s"],
    ),
    "slow_traffic": (
        "Медленное движение на подходе",
        [
            "speed_mean_5m",
            "speed_mean_15m",
            "speed_max_5m",
            "stopped_share_10m",
            "eta_dev_s",
            "dist_to_target_m",
            "gps_dist_ahead_m",
            "gps_stops_ahead",
        ],
    ),
    "long_dwell": (
        "Долгая стоянка или посадка",
        ["standing_s", "dwell_last_s", "dwell_mean3_s"],
    ),
    "chronic_segment": (
        "Проблемный участок: на прошлом круге здесь опаздывал",
        ["prev_loop_dev_s", "prev_loop_gain_s", "prev_loop_age_s"],
    ),
    "timetable": (
        "Особенность расписания и время суток",
        [
            "lead_s",
            "hour",
            "since_last_plan_s",
            "stops_ahead",
            "plan_dist_ahead_m",
            "plan_run_s",
            "last_trip_frac",
            "target_seq",
            "target_trip_frac",
            "trip_stops",
            "trip_no",
            "target_lat",
            "target_lon",
            VEHICLE_COLUMN,
        ],
    ),
    "signal": (
        "Нестабильный сигнал GPS",
        ["tel_age_s", "fixes_15m", "ok_share_15m", "gps_found_share", "route_offset_m"],
    ),
}


def reason_contributions(model: DelayModel, table: pd.DataFrame, fast: bool = True) -> pd.DataFrame:
    """Seconds each reason adds to the predicted delay, one column per reason.

    The columns plus ``expected_s`` sum up to ``model.predict(table)["delay_s"]``.
    ``fast`` uses CatBoost's approximate SHAP: several times faster, the sum is still exact,
    the split between features is a bit rougher. Scoring many rows in one call is much
    cheaper per row than one by one.
    """
    X = make_matrix(table, model.columns)
    categorical = [c for c in model.columns if c == VEHICLE_COLUMN]
    shap = model.regressor.get_feature_importance(
        Pool(X, cat_features=categorical),
        type="ShapValues",
        shap_calc_type="Approximate" if fast else "Regular",
    )
    by_feature = pd.DataFrame(shap[:, :-1], columns=model.columns, index=table.index)

    out = pd.DataFrame(index=table.index)
    for key, (_, features) in REASONS.items():
        present = [f for f in features if f in by_feature.columns]
        out[key] = by_feature[present].sum(axis=1)
    out["expected_s"] = shap[:, -1]

    if model.lines is not None:
        w = model.baseline_weight
        a, b = model.lines[False]
        a_layover, b_layover = model.lines[True]
        cur = table["cur_dev_s"].fillna(0.0).to_numpy()
        layover = table["new_trip_ahead"].to_numpy() > 0
        out["carried_delay"] += w * a * cur
        out["layover"] += w * np.where(layover, (a_layover - a) * cur + (b_layover - b), 0.0)
        out["expected_s"] += w * b
    return out


def main_reason(contributions: pd.DataFrame) -> pd.DataFrame:
    """The reason that pushes the most towards being late, with its text and seconds.

    If nothing pushes towards late (the bus is expected early or on time), the reason that
    pulls the most towards early is given instead.
    """
    keys = list(REASONS)
    values = contributions[keys].to_numpy()
    late = values.max(axis=1) > 0
    index = np.where(late, values.argmax(axis=1), values.argmin(axis=1))
    reason = [keys[i] for i in index]
    return pd.DataFrame(
        {
            "reason": reason,
            "reason_text": [REASONS[key][0] for key in reason],
            "reason_s": values[np.arange(len(values)), index],
        },
        index=contributions.index,
    )
