"""Client for predictor's ``POST /predict``.

Every way the call can go wrong — no URL configured, predictor down,
slow, answering with an error or with something that does not match the
batch — surfaces as one ``PredictorError``: the caller's reaction is the
same fallback in each case.
"""

import asyncio
from collections.abc import Sequence

import httpx
from pydantic import TypeAdapter, ValidationError

from contracts import PredictRequest, PredictResponse

_REQUESTS = TypeAdapter(list[PredictRequest])
_RESPONSES = TypeAdapter(list[PredictResponse])


class PredictorError(Exception):
    """predictor gave no usable answer to a batch."""


class PredictorClient:
    """One HTTP client for the process; ``base_url=None`` means no predictor configured."""

    def __init__(
        self,
        base_url: str | None,
        timeout_s: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._client = (
            httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)
            if base_url
            else None
        )

    async def predict(self, requests: Sequence[PredictRequest]) -> list[PredictResponse]:
        """Send one batch and return the answers in request order.

        The whole call, connecting included, is capped at ``timeout_s``.
        Answers must match the requests one to one and in the same order.
        """
        if self._client is None:
            raise PredictorError("predictor_url is not set")
        body = _REQUESTS.dump_json(list(requests))
        try:
            async with asyncio.timeout(self._timeout_s):
                response = await self._client.post(
                    "/predict", content=body, headers={"content-type": "application/json"}
                )
            response.raise_for_status()
            answers = _RESPONSES.validate_json(response.content)
        except (TimeoutError, httpx.HTTPError, ValidationError) as exc:
            raise PredictorError(f"{type(exc).__name__}: {exc}") from exc
        sent = [r.sample_id for r in requests]
        got = [a.sample_id for a in answers]
        if got != sent:
            raise PredictorError(
                f"answers do not match the batch: sent {len(sent)}, got {len(got)} "
                "or a different order"
            )
        return answers

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if self._client is not None:
            await self._client.aclose()
