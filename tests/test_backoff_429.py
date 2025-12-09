"""Tests that 429 handling uses base_backoff and succeeds on retry."""
# ruff: noqa: I001

from unittest.mock import Mock
import asyncio

import httpx
import pytest

from chat_limiter import BatchConfig
from chat_limiter.batch import create_chat_completion_requests, process_chat_completion_batch


@pytest.mark.asyncio
async def test_429_triggers_base_backoff_and_recovers_on_retry(openai_limiter, mock_openai_response, monkeypatch):
    # Record all sleep durations invoked during the test
    recorded_sleeps: list[float] = []

    async def fake_sleep(duration: float) -> None:
        recorded_sleeps.append(float(duration))
        return None

    # Monkeypatch asyncio.sleep globally so both limiter and batch sleeps are captured
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # Create a 429 response WITHOUT a Retry-After header (forces base_backoff path)
    rate_limited = Mock(spec=httpx.Response)
    rate_limited.status_code = 429
    rate_limited.headers = {"content-type": "application/json"}  # no retry-after
    rate_limited.json.return_value = {
        "error": {
            "message": "Rate limit exceeded. Try again later.",
            "type": "rate_limit_exceeded",
            "param": None,
            "code": "rate_limit_exceeded",
        }
    }
    rate_limited.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 Rate limit", request=Mock(), response=rate_limited
    )

    # First call returns 429, second call succeeds
    openai_limiter.async_client.request.side_effect = [rate_limited, mock_openai_response]

    # Prepare a simple request (OpenAI cheap model)
    requests = create_chat_completion_requests(
        model="gpt-3.5-turbo",
        prompts=["Hello"],
        max_tokens=16,
        temperature=0.0,
    )

    # Process via batch to exercise retry-after failure handling and a second attempt
    config = BatchConfig(show_progress=False)

    async with openai_limiter as limiter:
        results = await process_chat_completion_batch(limiter, requests, config)

    # One item should succeed on the second attempt
    assert len(results) == 1
    assert results[0].success is True
    assert openai_limiter.async_client.request.call_count == 2

    # Verify that limiter's 429 backoff used base_backoff=1.0:
    # backoff = base_backoff * (2 ** consecutive_errors) = 1.0 * 2 = 2.0
    assert recorded_sleeps, "No sleeps were recorded; expected limiter backoff to sleep."
    assert any(pytest.approx(s, rel=0.01) == 2.0 for s in recorded_sleeps)

