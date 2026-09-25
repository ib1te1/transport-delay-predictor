from fastapi import FastAPI

app = FastAPI(title="matcher")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "matcher"}
