"""Delay prediction for city buses, 10-15 minutes ahead.

Modules:

* :mod:`busdelay.data` - reading the organisers' csv files
* :mod:`busdelay.schedule` - per-vehicle timetable, trips and distances
* :mod:`busdelay.arrivals` - stop arrivals recovered from GPS
* :mod:`busdelay.features` - features of one forecast point
* :mod:`busdelay.folds` - cross-validation folds
* :mod:`busdelay.baselines` - simple predictors to compare the model with
* :mod:`busdelay.metrics` - MAE and the platform score
* :mod:`busdelay.model` - CatBoost regressor and "late" classifier
* :mod:`busdelay.explain` - reasons for the dispatcher from SHAP values
* :mod:`busdelay.inference` - forecasts for requests that carry their plan and telemetry
* :mod:`busdelay.submission` - writing and checking submission.csv
* :mod:`busdelay.cli` - ``python -m busdelay`` commands
"""

__version__ = "0.3.0"
