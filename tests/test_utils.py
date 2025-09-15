"""Tests for utilities handling event loop contexts."""

import importlib
import pytest
from unittest.mock import patch

utils = importlib.import_module("chat_limiter.utils")


async def _sample_coro(value: str) -> str:
    return value


def test_run_coro_blocking_regular_context_uses_asyncio_run():
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = "mocked"
        result = utils.run_coro_blocking(_sample_coro("real"))
        assert result == "mocked"
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_run_coro_blocking_in_running_loop_spawns_thread_and_returns_value():
    # In a running loop, it should still return the coroutine's result
    result = utils.run_coro_blocking(_sample_coro("ok"))
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_coro_blocking_reuses_single_executor(monkeypatch):
    """Ensure repeated calls in a running loop don't spawn new threads each time."""
    init_calls: list[int] = []

    class _DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    import concurrent.futures as _cf
    orig_executor = _cf.ThreadPoolExecutor

    class _DummyExecutor:
        def __init__(self, *_unused_args, **_unused_kwargs):
            init_calls.append(1)
            self._inner = orig_executor(max_workers=1)

        def submit(self, fn, *args, **kwargs):
            return self._inner.submit(fn, *args, **kwargs)

        # For context-manager compatibility in case code still uses `with ...` somewhere
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    # Reset singleton before patching
    # Clear lru_cache to force new executor creation
    utils._get_background_executor.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _DummyExecutor)

    # Two calls within a running loop context
    a = utils.run_coro_blocking(_sample_coro("A"))
    b = utils.run_coro_blocking(_sample_coro("B"))

    assert a == "A"
    assert b == "B"

    # Expect exactly one executor initialization if implementation reuses a singleton
    assert sum(init_calls) == 1


@pytest.mark.asyncio
async def test_run_coro_blocking_thread_creation_failure_propagates(monkeypatch):
    """If thread creation fails, the error should surface clearly (no nested asyncio errors)."""

    # Fail before any thread usage by replacing the accessor
    def _raise_exec():  # type: ignore[no-redef]
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("chat_limiter.utils._get_background_executor", _raise_exec)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        _ = utils.run_coro_blocking(_sample_coro("x"))

