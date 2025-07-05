"""Pytest configuration and fixtures for chat-limiter tests."""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from chat_limiter import ChatLimiter, Provider, ProviderConfig


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "x-ratelimit-limit-requests": "500",
        "x-ratelimit-remaining-requests": "499",
        "x-ratelimit-reset-requests": "60",
        "x-ratelimit-limit-tokens": "30000",
        "x-ratelimit-remaining-tokens": "29500",
        "x-ratelimit-reset-tokens": "60",
        "content-type": "application/json",
    }
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    return mock_response


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "anthropic-ratelimit-requests-remaining": "59",
        "anthropic-ratelimit-tokens-limit": "1000000",
        "anthropic-ratelimit-tokens-reset": "60",
        "content-type": "application/json",
    }
    mock_response.json.return_value = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    return mock_response


@pytest.fixture
def mock_openrouter_response():
    """Mock OpenRouter API response."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "content-type": "application/json",
    }
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openai/gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    return mock_response


@pytest.fixture
def mock_rate_limit_response():
    """Mock rate limit (429) response."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.headers = {
        "retry-after": "60",
        "content-type": "application/json",
    }
    mock_response.json.return_value = {
        "error": {
            "message": "Rate limit exceeded. Try again later.",
            "type": "rate_limit_exceeded",
            "param": None,
            "code": "rate_limit_exceeded",
        }
    }
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Rate limit exceeded", request=Mock(), response=mock_response
    )
    return mock_response


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "temperature": 0.7,
    }


@pytest.fixture
def sample_batch_requests():
    """Sample batch of chat completion requests."""
    return [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Question {i}"}],
            "max_tokens": 50,
        }
        for i in range(5)
    ]


@pytest.fixture
async def mock_async_client():
    """Mock async HTTP client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def mock_sync_client():
    """Mock sync HTTP client."""
    client = Mock(spec=httpx.Client)
    client.close = Mock()
    return client


@pytest.fixture
def openai_config():
    """OpenAI provider configuration."""
    return ProviderConfig(
        provider=Provider.OPENAI,
        base_url="https://api.openai.com/v1",
        request_limit_header="x-ratelimit-limit-requests",
        request_remaining_header="x-ratelimit-remaining-requests",
        request_reset_header="x-ratelimit-reset-requests",
        token_limit_header="x-ratelimit-limit-tokens",
        token_remaining_header="x-ratelimit-remaining-tokens",
        token_reset_header="x-ratelimit-reset-tokens",
        retry_after_header="retry-after",
        default_request_limit=500,
        default_token_limit=30000,
    )


@pytest.fixture
def anthropic_config():
    """Anthropic provider configuration."""
    return ProviderConfig(
        provider=Provider.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        request_remaining_header="anthropic-ratelimit-requests-remaining",
        token_limit_header="anthropic-ratelimit-tokens-limit",
        token_reset_header="anthropic-ratelimit-tokens-reset",
        retry_after_header="retry-after",
        default_request_limit=60,
        default_token_limit=1000000,
    )


@pytest.fixture
def openrouter_config():
    """OpenRouter provider configuration."""
    return ProviderConfig(
        provider=Provider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        auth_endpoint="https://openrouter.ai/api/v1/auth/key",
        default_request_limit=20,
        default_token_limit=1000000,
    )


@pytest.fixture
async def openai_limiter(mock_async_client, mock_sync_client, openai_config):
    """ChatLimiter configured for OpenAI with mocked clients."""
    limiter = ChatLimiter(
        config=openai_config,
        api_key="sk-test-key",
        http_client=mock_async_client,
        sync_http_client=mock_sync_client,
    )
    return limiter


@pytest.fixture
async def anthropic_limiter(mock_async_client, mock_sync_client, anthropic_config):
    """ChatLimiter configured for Anthropic with mocked clients."""
    limiter = ChatLimiter(
        config=anthropic_config,
        api_key="sk-ant-test-key",
        http_client=mock_async_client,
        sync_http_client=mock_sync_client,
    )
    return limiter


@pytest.fixture
async def openrouter_limiter(mock_async_client, mock_sync_client, openrouter_config):
    """ChatLimiter configured for OpenRouter with mocked clients."""
    limiter = ChatLimiter(
        config=openrouter_config,
        api_key="sk-or-test-key",
        http_client=mock_async_client,
        sync_http_client=mock_sync_client,
    )
    return limiter


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
