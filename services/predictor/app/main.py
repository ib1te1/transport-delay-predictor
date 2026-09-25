"""ML core service. Until a trained model is in place it answers with the
organizers' baseline: the predicted delay equals the current deviation.
When MODEL_DIR is absent or empty the service keeps answering with this
baseline, so the system starts without a trained model."""

from fastapi import FastAPI

from contracts import PredictRequest, PredictResponse, ReasonCode

app = FastAPI(title="predictor")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "predictor"}


@app.post("/predict")
def predict(requests: list[PredictRequest]) -> list[PredictResponse]:
    """Predict the delay at each request's target stop, in request order."""
    return [
        PredictResponse(
            sample_id=r.sample_id,
            prediction_s=r.cur_dev_s if r.cur_dev_s is not None else 0.0,
            p_late=None,
            reasons=[ReasonCode.unknown],
            model_version="baseline",
        )
        for r in requests
    ]
