"""ML core service: ``POST /predict`` answers the api with the delay model.

``GET /model`` says which model answers, ``GET /metrics`` how long the answers take.

The model is loaded once at startup from ``MODEL_DIR``. When the directory is absent, empty
or the model does not load, the service answers with the organizers' baseline (the
predicted delay equals the current deviation), so the system starts without a trained
model and a broken one does not take it down.
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

from app.config import PredictorConfig, PredictorSettings
from app.examples import bus_request
from app.metrics import Metrics, PredictTimings
from app.predictor import BaselinePredictor, ModelPredictor, Predictor, load_predictor
from common.config import load_section
from contracts import PredictRequest, PredictResponse

# The schema's own example for /predict gets only the baseline: its target stop is not ahead
# of T. This one is a made-up bus the model answers for.
PREDICT_EXAMPLES = {
    "late_bus": {
        "summary": "A bus 4 minutes late",
        "description": "40 minutes into the trip, a fix every 30 s, no cur_dev_s from matcher.",
        "value": [bus_request(1, 240.0, step_s=30).model_dump(mode="json")],
    }
}


class ModelInfo(BaseModel):
    """What answers /predict, for the dashboard and for a quick check by hand."""

    model_version: str
    kind: str
    cur_dev_source: str | None = None
    note: str | None = None
    created: str | None = None
    # MAE in seconds on the model's cross-validation, see docs/specs/ml-model.md
    cv_mae_s: dict[str, float] = {}


def log_to_stderr() -> None:
    """uvicorn sets up only its own loggers; without this the service's INFO lines, such as
    which model was loaded, are lost."""
    log = logging.getLogger("app")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)


def create_app(settings: PredictorSettings | None = None) -> FastAPI:
    """The service; ``settings`` replace the environment in tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log_to_stderr()
        current = settings or PredictorSettings()
        config = load_section(current.config_path, "predictor", PredictorConfig)
        app.state.predictor = load_predictor(current.model_dir, config)
        app.state.config = config
        yield

    app = FastAPI(title="predictor", lifespan=lifespan)
    started = datetime.now(UTC)
    timings = PredictTimings()

    def predictor_of(request: Request) -> Predictor:
        # without the lifespan (a bare TestClient) there is no model: the baseline answers
        return getattr(request.app.state, "predictor", None) or BaselinePredictor()

    @app.middleware("http")
    async def time_predict(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # here and not in the handler, so parsing the body counts too
        if request.url.path != "/predict":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        timings.add(
            (time.perf_counter() - start) * 1000,
            ok=response.status_code < 400,
            batch=getattr(request.state, "batch", 0),
            body_bytes=int(request.headers.get("content-length", 0)),
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness for the compose healthcheck."""
        return {"status": "ok", "service": "predictor"}

    @app.post("/predict")
    def predict(requests: list[PredictRequest], request: Request) -> list[PredictResponse]:
        """Predict the delay at each request's target stop, in request order."""
        request.state.batch = len(requests)
        return predictor_of(request).predict(requests)

    @app.get("/metrics")
    def metrics(request: Request) -> Metrics:
        """How long /predict takes over the last calls, and which model answers it."""
        return Metrics(
            model_version=predictor_of(request).version,
            started=started,
            predict=timings.summary(),
        )

    @app.get("/model")
    def model(request: Request) -> ModelInfo:
        """The model behind /predict: version, where cur_dev_s comes from, CV quality."""
        predictor = predictor_of(request)
        if isinstance(predictor, ModelPredictor):
            meta = predictor.model.meta
            cv = meta.get("metrics", {}).get("cv", {})
            return ModelInfo(
                model_version=predictor.version,
                kind="model",
                cur_dev_source=request.app.state.config.cur_dev_source,
                created=meta.get("created"),
                cv_mae_s={name: round(row["mae"], 1) for name, row in cv.items()},
            )
        return ModelInfo(
            model_version=predictor.version,
            kind="baseline",
            note=getattr(predictor, "why", None),
        )

    fastapi_openapi = app.openapi

    def openapi_with_example() -> dict[str, Any]:
        """FastAPI's schema plus a ready batch for "Try it out" on ``/predict``, built once.

        The example goes in after FastAPI: it drops null values from examples, and the
        contract requires ``cur_dev_s`` and ``time_fact`` even when they are null.
        """
        if app.openapi_schema is None:
            schema = fastapi_openapi()
            body = schema["paths"]["/predict"]["post"]["requestBody"]["content"]
            body["application/json"]["examples"] = PREDICT_EXAMPLES
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi_with_example
    return app


app = create_app()
