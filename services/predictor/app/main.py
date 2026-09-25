from fastapi import FastAPI

app = FastAPI(title="predictor")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "predictor"}
