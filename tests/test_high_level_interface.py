"""
Tests for the high-level chat completion interface.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from chat_limiter import ChatLimiter, Message, MessageRole, Provider


class TestChatLimiterForModel:
    def test_for_model_openai(self):
        """Test ChatLimiter.for_model with OpenAI model."""
        limiter = ChatLimiter.for_model("gpt-4o", "sk-test-key")
        assert limiter.provider == Provider.OPENAI
        assert limiter.api_key == "sk-test-key"

    def test_for_model_anthropic(self):
        """Test ChatLimiter.for_model with Anthropic model."""
        limiter = ChatLimiter.for_model("claude-3-5-sonnet-20241022", "sk-ant-test")
        assert limiter.provider == Provider.ANTHROPIC
        assert limiter.api_key == "sk-ant-test"

    def test_for_model_openrouter(self):
        """Test ChatLimiter.for_model with OpenRouter model."""
        limiter = ChatLimiter.for_model("openai/gpt-4o", "sk-or-test")
        assert limiter.provider == Provider.OPENROUTER
        assert limiter.api_key == "sk-or-test"

    def test_for_model_unknown(self):
        """Test ChatLimiter.for_model with unknown model."""
        with pytest.raises(ValueError, match="Could not determine provider"):
            ChatLimiter.for_model("unknown-model", "test-key")

    def test_for_model_with_kwargs(self):
        """Test ChatLimiter.for_model with additional kwargs."""
        limiter = ChatLimiter.for_model(
            "gpt-4o",
            "sk-test-key",
            enable_adaptive_limits=False
        )
        assert limiter.provider == Provider.OPENAI
        assert not limiter.enable_adaptive_limits

    def test_for_model_with_provider_override(self):
        """Test ChatLimiter.for_model with provider override."""
        # Test string provider override
        limiter = ChatLimiter.for_model(
            "custom-model",
            "sk-test-key",
            provider="openai"
        )
        assert limiter.provider == Provider.OPENAI

        # Test Provider enum override
        limiter = ChatLimiter.for_model(
            "custom-model",
            "sk-test-key",
            provider=Provider.ANTHROPIC
        )
        assert limiter.provider == Provider.ANTHROPIC

    def test_for_model_with_env_api_key(self):
        """Test ChatLimiter.for_model with environment variable API key."""
        import os

        # Set environment variable
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-env-key"

        try:
            limiter = ChatLimiter.for_model("gpt-4o")
            assert limiter.provider == Provider.OPENAI
            assert limiter.api_key == "test-env-key"
        finally:
            # Clean up
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_for_model_missing_env_key(self):
        """Test ChatLimiter.for_model with missing environment variable."""
        import os

        # Ensure environment variable is not set
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not set"):
                ChatLimiter.for_model("claude-3-sonnet-20240229")
        finally:
            # Clean up
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key


class TestChatCompletionAsync:
    @pytest.fixture
    def mock_limiter(self):
        """Create a ChatLimiter with mocked HTTP client."""
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        # Mock headers as a dict-like object
        mock_response.headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }

        # Create mock HTTP client
        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.aclose = AsyncMock()

        # Create limiter with mock client
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            http_client=mock_client
        )

        return limiter

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, mock_limiter):
        """Test basic chat completion."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        async with mock_limiter as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=messages
            )

        assert response.id == "chatcmpl-test"
        assert response.model == "gpt-4o-2024-08-06"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello there!"
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self, mock_limiter):
        """Test chat completion with all parameters."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        async with mock_limiter as limiter:
            await limiter.chat_completion(
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop=["\\n"],
                stream=False,
                frequency_penalty=0.5,
                presence_penalty=0.3,
            )

        # Verify the request was made to the underlying HTTP client
        mock_limiter.async_client.request.assert_called_once()
        call_args = mock_limiter.async_client.request.call_args

        assert call_args[0][0] == "POST"  # method
        assert call_args[0][1] == "/chat/completions"  # url

        # Check JSON payload
        json_data = call_args[1]["json"]
        assert json_data["model"] == "gpt-4o"
        assert json_data["max_tokens"] == 100
        assert json_data["temperature"] == 0.7
        assert json_data["frequency_penalty"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_completion_without_context(self):
        """Test chat completion without context manager."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with pytest.raises(RuntimeError, match="async context manager"):
            await limiter.chat_completion("gpt-4o", messages)

    @pytest.mark.asyncio
    async def test_simple_chat(self, mock_limiter):
        """Test simple chat method."""
        async with mock_limiter as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o",
                prompt="Hello!",
                max_tokens=50
            )

        assert response == "Hello there!"

    @pytest.mark.asyncio
    async def test_simple_chat_empty_response(self, mock_limiter):
        """Test simple chat with empty response."""
        # Mock empty response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "choices": []
        }
        # Mock headers as a dict-like object
        mock_response.headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_limiter.async_client.request.return_value = mock_response

        async with mock_limiter as limiter:
            response = await limiter.simple_chat("gpt-4o", "Hello!")

        assert response == ""


class TestChatCompletionSync:
    @pytest.fixture
    def mock_sync_limiter(self):
        """Create a ChatLimiter with mocked sync HTTP client."""
        # Create mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        # Mock headers as a dict-like object
        mock_response.headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }

        # Create mock HTTP client
        mock_client = Mock()
        mock_client.request.return_value = mock_response
        mock_client.close = Mock()

        # Create limiter with mock client
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            sync_http_client=mock_client
        )

        return limiter

    def test_chat_completion_sync_basic(self, mock_sync_limiter):
        """Test basic synchronous chat completion."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with mock_sync_limiter as limiter:
            response = limiter.chat_completion_sync(
                model="gpt-4o",
                messages=messages
            )

        assert response.id == "chatcmpl-test"
        assert response.choices[0].message.content == "Hello there!"

    def test_chat_completion_sync_without_context(self):
        """Test sync chat completion without context manager."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with pytest.raises(RuntimeError, match="sync context manager"):
            limiter.chat_completion_sync("gpt-4o", messages)

    def test_simple_chat_sync(self, mock_sync_limiter):
        """Test simple synchronous chat method."""
        with mock_sync_limiter as limiter:
            response = limiter.simple_chat_sync(
                model="gpt-4o",
                prompt="Hello!",
                max_tokens=50
            )

        assert response == "Hello there!"


class TestChatCompletionWithDifferentProviders:
    def test_anthropic_adapter_integration(self):
        """Test that Anthropic adapter is used correctly."""
        limiter = ChatLimiter(provider=Provider.ANTHROPIC, api_key="sk-ant-test")

        # Mock the request method to verify the adapter is called
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "msg_test",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 5, "output_tokens": 3}
        }

        limiter.request_sync = Mock(return_value=mock_response)

        messages = [Message(role=MessageRole.USER, content="Test")]

        with limiter:
            response = limiter.chat_completion_sync("claude-3-sonnet-20240229", messages)

        # Verify the request was made with Anthropic format
        limiter.request_sync.assert_called_once()
        call_args = limiter.request_sync.call_args

        assert call_args[0][0] == "POST"  # method
        assert call_args[0][1] == "/messages"  # Anthropic endpoint

        # Check that the response was parsed correctly
        assert response.provider == "anthropic"
        assert response.choices[0].message.content == "Hello!"

    def test_openrouter_adapter_integration(self):
        """Test that OpenRouter adapter is used correctly."""
        limiter = ChatLimiter(provider=Provider.OPENROUTER, api_key="sk-or-test")

        # Mock the request method
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}]
        }

        limiter.request_sync = Mock(return_value=mock_response)

        messages = [Message(role=MessageRole.USER, content="Test")]

        with limiter:
            response = limiter.chat_completion_sync("openai/gpt-4o", messages)

        # Verify the request was made
        limiter.request_sync.assert_called_once()
        call_args = limiter.request_sync.call_args

        assert call_args[0][1] == "/chat/completions"  # OpenRouter endpoint

        # Check that the response was parsed correctly
        assert response.provider == "openrouter"
