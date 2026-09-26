import asyncio

import pytest

import app.main as main_module


@pytest.fixture
def no_prediction_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the background prediction loop with a no-op that waits for cancellation.

    Without this, entering the lifespan runs the real loop against the
    shared dev Redis and Postgres, scoring against the default streams
    instead of the unique ones tests are supposed to use.
    """

    async def wait_until_cancelled(*args, **kwargs) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(main_module, "run_prediction_loop", wait_until_cancelled)
