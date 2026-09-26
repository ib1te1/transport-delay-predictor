import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app.config import SeedPeriodConfig, load_ingest_config, to_utc
from app.tcp_server import run_ndtp_server
from common.config import ServiceSettings, load_section
from common.db import make_pool

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = ServiceSettings()
    config, dataset = load_ingest_config(settings.config_path)
    task: asyncio.Task | None = None
    pool = None
    if config.mode == "emulator":
        seed = load_section(settings.config_path, "seed", SeedPeriodConfig)
        assert config.ndtp.dataset_anchor is not None
        anchor = to_utc(config.ndtp.dataset_anchor, dataset.zone)
        pool = make_pool(settings.database_url)
        pool.open(wait=False)

        async def supervise() -> None:
            delay = 1.0
            while True:
                try:
                    await run_ndtp_server(
                        pool,
                        settings.redis_url,
                        config.ndtp,
                        period=seed.period,
                        dataset_anchor=anchor,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("NDTP listener failed; restarting in %.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

        task = asyncio.create_task(supervise())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if pool is not None:
            await asyncio.to_thread(pool.close)


app = FastAPI(title="ingest", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingest"}
