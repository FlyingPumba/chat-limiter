"""Context-specific tests for provider detection sync wrappers."""

import pytest
from unittest.mock import patch

from chat_limiter.models import (
    detect_provider_from_model_async,
    detect_provider_from_model_sync,
    ModelDiscoveryResult,
)


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


