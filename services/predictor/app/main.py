"""ML core service: ``POST /predict`` answers the api with the delay model.

The model is loaded once at startup from ``MODEL_DIR``. When the directory is absent, empty
or the model does not load, the service answers with the organizers' baseline (the
predicted delay equals the current deviation), so the system starts without a trained
model and a broken one does not take it down.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.config import PredictorConfig, PredictorSettings
from app.predictor import BaselinePredictor, ModelPredictor, Predictor, load_predictor
from common.config import load_section
from contracts import PredictRequest, PredictResponse


class ModelInfo(BaseModel):
    """What answers /predict, for the dashboard and for a quick check by hand."""

    model_version: str
    kind: str
    cur_dev_source: str | None = None
    note: str | None = None
    created: str | None = None
    # MAE in seconds on the model's cross-validation, see docs/specs/ml-model.md
    cv_mae_s: dict[str, float] = {}


def create_app(settings: PredictorSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        current = settings or PredictorSettings()
        config = load_section(current.config_path, "predictor", PredictorConfig)
        app.state.predictor = load_predictor(current.model_dir, config)
        app.state.config = config
        yield

    app = FastAPI(title="predictor", lifespan=lifespan)

    def predictor_of(request: Request) -> Predictor:
        # without the lifespan (a bare TestClient) there is no model: the baseline answers
        return getattr(request.app.state, "predictor", None) or BaselinePredictor()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "predictor"}

    @app.post("/predict")
    def predict(requests: list[PredictRequest], request: Request) -> list[PredictResponse]:
        """Predict the delay at each request's target stop, in request order."""
        return predictor_of(request).predict(requests)

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

    return app


app = create_app()
