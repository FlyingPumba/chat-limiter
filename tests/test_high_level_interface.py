"""Tests for high-level interface functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from chat_limiter import ChatLimiter, Provider, Message, MessageRole


class TestChatLimiterForModel:
    """Tests for ChatLimiter.for_model class method."""

    def test_for_model_openai(self):
        """Test for_model with OpenAI model."""
        limiter = ChatLimiter.for_model("gpt-4o", api_key="sk-test", use_dynamic_discovery=False)
        assert limiter.provider == Provider.OPENAI

    def test_for_model_anthropic(self):
        """Test for_model with Anthropic model."""
        limiter = ChatLimiter.for_model("claude-3-sonnet-20240229", api_key="sk-ant-test", use_dynamic_discovery=False)
        assert limiter.provider == Provider.ANTHROPIC

    def test_for_model_openrouter(self):
        """Test for_model with OpenRouter model."""
        limiter = ChatLimiter.for_model("meta-llama/llama-3.1-405b-instruct", api_key="sk-or-test", use_dynamic_discovery=False)
        assert limiter.provider == Provider.OPENROUTER

    def test_for_model_openai_with_prefix(self):
        """Test for_model with openai/ prefix routes to OpenAI when base model exists."""
        limiter = ChatLimiter.for_model("openai/gpt-4o", api_key="sk-test", use_dynamic_discovery=False)
        assert limiter.provider == Provider.OPENAI

    def test_for_model_anthropic_with_prefix(self):
        """Test for_model with anthropic/ prefix routes to Anthropic when base model exists."""
        limiter = ChatLimiter.for_model("anthropic/claude-3-sonnet-20240229", api_key="sk-ant-test", use_dynamic_discovery=False)
        assert limiter.provider == Provider.ANTHROPIC

    def test_for_model_unknown(self):
        """Test for_model with unknown model."""
        with pytest.raises(ValueError, match="Could not determine provider"):
            ChatLimiter.for_model("unknown-model", api_key="test", use_dynamic_discovery=False)

    def test_for_model_with_kwargs(self):
        """Test for_model with additional arguments."""
        limiter = ChatLimiter.for_model(
            "gpt-4o", 
            api_key="sk-test", 
            timeout=30, 
            max_retries=5
        )
        assert limiter.provider == Provider.OPENAI
        assert limiter._user_timeout == 30
        assert limiter._user_max_retries == 5

    def test_for_model_with_provider_override(self):
        """Test for_model with provider override."""
        limiter = ChatLimiter.for_model(
            "custom-model", 
            api_key="test", 
            provider=Provider.OPENAI
        )
        assert limiter.provider == Provider.OPENAI

    def test_for_model_with_env_api_key(self):
        """Test for_model with environment variable API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-test"}):
            limiter = ChatLimiter.for_model("gpt-4o", use_dynamic_discovery=False)
            assert limiter.provider == Provider.OPENAI

    def test_for_model_missing_env_key(self):
        """Test for_model without API key raises appropriate error."""
        # Clear environment variables to ensure no API keys are available
        import os
        original_env = {}
        env_vars_to_clear = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
        
        # Save original values and clear them
        for var in env_vars_to_clear:
            original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]
        
        try:
            with pytest.raises(ValueError, match="API key"):
                ChatLimiter.for_model("gpt-4o", use_dynamic_discovery=False)
        finally:
            # Restore original values
            for var, value in original_env.items():
                if value is not None:
                    os.environ[var] = value


class TestChatCompletionAsync:
    """Tests for async chat completion functionality."""

    @pytest.fixture
    def mock_limiter(self):
        """Mock ChatLimiter for testing."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        }
        mock_client.request.return_value = mock_response

        return ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            http_client=mock_client
        )

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, mock_limiter):
        """Test basic chat completion."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        async with mock_limiter as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=messages
            )

        assert response.success
        assert response.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self, mock_limiter):
        """Test chat completion with additional parameters."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        async with mock_limiter as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )

        assert response.success
        assert response.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_completion_without_context(self):
        """Test chat completion without context manager raises error."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with pytest.raises(RuntimeError, match="must be used as an async context manager"):
            await limiter.chat_completion(model="gpt-4o", messages=messages)

    @pytest.mark.asyncio
    async def test_simple_chat(self, mock_limiter):
        """Test simple chat convenience method."""
        async with mock_limiter as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o",
                prompt="Hello!",
                max_tokens=50
            )

        assert response == "Hello!"

    @pytest.mark.asyncio
    async def test_simple_chat_empty_response(self, mock_limiter):
        """Test simple chat with empty response."""
        # Mock empty response
        mock_limiter.async_client.request.return_value.json.return_value = {
            "choices": []
        }

        async with mock_limiter as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o",
                prompt="Hello!"
            )

        assert response == ""


class TestChatCompletionSync:
    """Tests for sync chat completion functionality."""

    @pytest.fixture
    def mock_limiter(self):
        """Mock ChatLimiter for testing."""
        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        }
        mock_client.request.return_value = mock_response

        return ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            sync_http_client=mock_client
        )

    def test_chat_completion_sync_basic(self, mock_limiter):
        """Test basic sync chat completion."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with mock_limiter as limiter:
            response = limiter.chat_completion_sync(
                model="gpt-4o",
                messages=messages
            )

        assert response.success
        assert response.choices[0].message.content == "Hello!"

    def test_chat_completion_sync_without_context(self):
        """Test sync chat completion without context manager raises error."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        messages = [Message(role=MessageRole.USER, content="Hello!")]

        with pytest.raises(RuntimeError, match="must be used as a sync context manager"):
            limiter.chat_completion_sync(model="gpt-4o", messages=messages)

    def test_simple_chat_sync(self, mock_limiter):
        """Test simple chat sync convenience method."""
        with mock_limiter as limiter:
            response = limiter.simple_chat_sync(
                model="gpt-4o",
                prompt="Hello!",
                max_tokens=50
            )

        assert response == "Hello!"