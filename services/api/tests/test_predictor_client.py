import asyncio
import json

import httpx
import pytest
from factories import at, predict_request

from app.predictor_client import PredictorClient, PredictorError
from contracts import PredictRequest

REQUESTS = [predict_request(7, at(0), cur_dev_s=60.0), predict_request(8, at(0))]


def answers_for(body: list[dict], prediction_s: float = 100.0) -> list[dict]:
    return [
        {
            "sample_id": r["sample_id"],
            "prediction_s": prediction_s,
            "p_late": 0.4,
            "reasons": ["accumulated_delay"],
            "model_version": "m1",
        }
        for r in body
    ]


def answering(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=answers_for(json.loads(request.content)))


def client(handler, timeout_s: float = 1.0) -> PredictorClient:
    return PredictorClient("http://predictor", timeout_s, transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_predict_posts_the_batch_and_returns_answers_in_order() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answering(request)

    predictor = client(handler)
    try:
        answers = await predictor.predict(REQUESTS)
    finally:
        await predictor.aclose()

    assert [a.sample_id for a in answers] == [r.sample_id for r in REQUESTS]
    assert (seen[0].method, seen[0].url.path) == ("POST", "/predict")
    sent = [PredictRequest.model_validate(item) for item in json.loads(seen[0].content)]
    assert sent == REQUESTS


def server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="boom")


def not_json(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="not json")


def refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def one_answer_short(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=answers_for(json.loads(request.content))[1:])


def reversed_order(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=answers_for(json.loads(request.content))[::-1])


@pytest.mark.anyio
@pytest.mark.parametrize(
    "handler",
    [server_error, not_json, refused, one_answer_short, reversed_order],
    ids=lambda h: h.__name__,
)
async def test_predict_raises_predictor_error_on_a_bad_answer(handler) -> None:
    predictor = client(handler)
    try:
        with pytest.raises(PredictorError):
            await predictor.predict(REQUESTS)
    finally:
        await predictor.aclose()


@pytest.mark.anyio
@pytest.mark.timeout(10)
async def test_predict_gives_up_after_the_timeout() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return answering(request)

    predictor = client(slow, timeout_s=0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with pytest.raises(PredictorError):
            await predictor.predict(REQUESTS)
    finally:
        await predictor.aclose()

    assert loop.time() - started < 1


@pytest.mark.anyio
async def test_predict_without_a_predictor_url_raises() -> None:
    predictor = PredictorClient(None, 1.0)

    with pytest.raises(PredictorError, match="predictor_url"):
        await predictor.predict(REQUESTS)
    await predictor.aclose()
