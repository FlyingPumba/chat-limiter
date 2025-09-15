"""Context-specific tests for provider detection sync wrappers."""

import pytest
from unittest.mock import patch

from chat_limiter.models import (
    detect_provider_from_model_sync,
    ModelDiscoveryResult,
)
from chat_limiter import ChatLimiter


@pytest.mark.asyncio
async def test_detect_provider_from_model_sync_in_running_loop_uses_thread_and_returns():
    # Mock the async function to avoid network
    with patch(
        "chat_limiter.models.detect_provider_from_model_async",
        autospec=True,
    ) as mock_async:
        mock_async.return_value = ModelDiscoveryResult(
            found_provider="openai", model_found=True
        )

        result = detect_provider_from_model_sync("gpt-4o", {"openai": "k"})
        assert result.found_provider == "openai"
        assert result.model_found is True


def test_detect_provider_from_model_sync_regular_context_uses_asyncio_run():
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = ModelDiscoveryResult(
            found_provider="openai", model_found=True
        )
        result = detect_provider_from_model_sync("gpt-4o", {"openai": "k"})
        assert result.found_provider == "openai"
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_detect_provider_from_model_sync_thread_failure_propagates(monkeypatch):
    # Force the bridge to fail before any thread is started
    def _raise_exec():  # type: ignore[no-redef]
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr("chat_limiter.utils._get_background_executor", _raise_exec)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        _ = detect_provider_from_model_sync("gpt-4o", {"openai": "k"})


@pytest.mark.asyncio
async def test_for_model_running_loop_no_discovery_does_not_use_threads(monkeypatch):
    # Make thread creation fail; should not be used when discovery disabled
    class _FailingExecutor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("should not be called")

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _FailingExecutor)

    limiter = ChatLimiter.for_model("gpt-4o", api_key="sk-test", use_dynamic_discovery=False)
    assert limiter.provider.value == "openai"


@pytest.mark.asyncio
async def test_for_model_running_loop_with_discovery_uses_thread_and_succeeds(monkeypatch):
    # Reset global executor in utils
    import importlib
    utils = importlib.import_module("chat_limiter.utils")
    utils._get_background_executor.cache_clear()  # type: ignore[attr-defined]

    init_calls: list[int] = []

    class _DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    import concurrent.futures as _cf
    orig_executor = _cf.ThreadPoolExecutor

    class _CountingExecutor:
        def __init__(self, *args, **kwargs):
            init_calls.append(1)
            self._inner = orig_executor(max_workers=1)

        def submit(self, fn, *args, **kwargs):
            return self._inner.submit(fn, *args, **kwargs)

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _CountingExecutor)

    # Mock dynamic discovery async function to avoid network
    with patch(
        "chat_limiter.models.detect_provider_from_model_async",
        autospec=True,
    ) as mock_async:
        mock_async.return_value = ModelDiscoveryResult(found_provider="openai", model_found=True)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            limiter = ChatLimiter.for_model("gpt-nonexistent", api_key="sk-env", use_dynamic_discovery=True)
            assert limiter.provider.value == "openai"

    assert sum(init_calls) == 1


def test_for_model_no_loop_with_discovery_uses_asyncio_run():
    # In non-async context, bridge should call asyncio.run
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = ModelDiscoveryResult(found_provider="openai", model_found=True)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            limiter = ChatLimiter.for_model("gpt-unknown-xyz", api_key="sk-env", use_dynamic_discovery=True)
            assert limiter.provider.value == "openai"
        assert mock_run.called

