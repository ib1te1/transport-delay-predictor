"""Continuous stop detection from the persisted ingest telemetry."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis import Redis

from app.config import MatcherConfig, load_matcher_settings
from app.store import Store
from common.config import ServiceSettings, load_section
from common.db import make_pool

log = logging.getLogger(__name__)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = settings or ServiceSettings()
        config = load_section(runtime.config_path, "matcher", MatcherConfig)
        thresholds = load_matcher_settings(runtime.config_path.parent / "assumptions.yaml")
        pool = make_pool(runtime.database_url)
        pool.open(wait=True)
        redis = Redis.from_url(runtime.redis_url)
        store = Store(pool, thresholds)
        app.state.store = store
        app.state.redis = redis

        async def consume() -> None:
            while True:
                try:
                    if await asyncio.to_thread(store.process_next):
                        continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("matcher input failed; retrying")
                await asyncio.sleep(config.poll_interval_sec)

        async def publish() -> None:
            while True:
                try:
                    if await asyncio.to_thread(store.publish_pending, redis):
                        continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("matcher event delivery failed; retrying")
                await asyncio.sleep(config.poll_interval_sec)

        tasks = [asyncio.create_task(consume()), asyncio.create_task(publish())]
        app.state.tasks = tasks
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            redis.close()
            pool.close()

    app = FastAPI(title="matcher", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "matcher"}

    @app.get("/ready")
    def ready(request: Request):
        if any(task.done() for task in request.app.state.tasks):
            return JSONResponse(status_code=503, content={"status": "worker_stopped"})
        try:
            request.app.state.store.ready()
            request.app.state.redis.ping()
        except Exception:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return {"status": "ready", "service": "matcher"}

    @app.get("/quality")
    def quality(request: Request) -> dict:
        return request.app.state.store.quality()

    return app


app = create_app()
