"""Tests for utilities handling event loop contexts."""

import asyncio
import pytest
from unittest.mock import patch

from chat_limiter.utils import run_coro_blocking


async def _sample_coro(value: str) -> str:
    return value


def test_run_coro_blocking_regular_context_uses_asyncio_run():
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = "mocked"
        result = run_coro_blocking(_sample_coro("real"))
        assert result == "mocked"
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_coro_blocking_in_running_loop_spawns_thread_and_returns_value():
    # In a running loop, it should still return the coroutine's result
    result = run_coro_blocking(_sample_coro("ok"))
    assert result == "ok"


