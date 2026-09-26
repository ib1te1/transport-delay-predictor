"""Continuous stop detection from the telemetry stream."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis import Redis

from app.config import MatcherConfig, load_matcher_settings
from app.store import StateMismatch, Store
from common.config import ServiceSettings, load_section
from common.db import make_pool

log = logging.getLogger(__name__)

MAX_RETRY_SEC = 10.0


def log_to_stderr() -> None:
    """uvicorn sets up only its own loggers; without this INFO lines of the service are lost."""
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_to_stderr()
        runtime = settings or ServiceSettings()
        config = load_section(runtime.config_path, "matcher", MatcherConfig)
        thresholds = load_matcher_settings(runtime.config_path.parent / "assumptions.yaml")
        pool = make_pool(runtime.database_url)
        redis = Redis.from_url(runtime.redis_url)
        store = Store(
            pool, redis, thresholds, batch_size=config.batch_size, block_ms=config.block_ms
        )
        app.state.store = store
        app.state.redis = redis
        app.state.tasks = []
        try:
            pool.open(wait=True)
            try:
                await asyncio.to_thread(store.load)
            except StateMismatch as exc:
                log.error("%s", exc)
                raise
            await asyncio.to_thread(store.check_stream)

            async def consume() -> None:
                delay = config.poll_interval_sec
                while True:
                    try:
                        # XREAD blocks while the stream is idle, so no sleep here.
                        await asyncio.to_thread(store.process_batch)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        store.failing = True
                        log.exception("matcher input failed; retrying in %.2f s", delay)
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, MAX_RETRY_SEC)
                        continue
                    store.failing = False
                    delay = config.poll_interval_sec

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
        finally:
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
        if request.app.state.store.failing:
            return JSONResponse(status_code=503, content={"status": "input_failing"})
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
