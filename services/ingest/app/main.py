from fastapi import FastAPI

app = FastAPI(title="ingest")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingest"}
