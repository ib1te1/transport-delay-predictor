import asyncio

import pytest

from app.main import keep_running


class Stop(Exception):
    pass


def test_worker_backs_off_flags_failures_and_recovers():
    failing = {"output": False}
    calls = 0
    waits = []

    def step():
        nonlocal calls
        calls += 1
        if calls <= 4:
            raise ConnectionError("redis down")
        return 0

    async def sleep(delay):
        waits.append((delay, failing["output"]))
        if len(waits) == 5:
            raise Stop

    with pytest.raises(Stop):
        asyncio.run(keep_running(step, failing, "output", 4, idle_wait=True, sleep=sleep))
    assert waits == [(4, True), (8, True), (10, True), (10, True), (4, False)]
