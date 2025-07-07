"""Tests for the main ChatLimiter class."""

from unittest.mock import patch

import pytest

from chat_limiter import ChatLimiter, LimiterState, Provider
from chat_limiter.providers import get_provider_config


class TestChatLimiterInitialization:
    """Tests for ChatLimiter initialization."""

    def test_init_with_provider(self):
        """Test initialization with provider."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        assert limiter.provider == Provider.OPENAI
        assert limiter.api_key == "sk-test"
        assert limiter.config.provider == Provider.OPENAI
        assert limiter.config.base_url == "https://api.openai.com/v1"

    def test_init_with_base_url(self):
        """Test initialization with base URL."""
        limiter = ChatLimiter(base_url="https://api.openai.com/v1", api_key="sk-test")

        assert limiter.provider == Provider.OPENAI
        assert limiter.config.base_url == "https://api.openai.com/v1"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = get_provider_config(Provider.ANTHROPIC)
        limiter = ChatLimiter(config=config, api_key="sk-test")

        assert limiter.provider == Provider.ANTHROPIC
        assert limiter.config == config

    def test_init_invalid_url(self):
        """Test initialization with invalid URL."""
        with pytest.raises(ValueError, match="Could not detect provider"):
            ChatLimiter(base_url="https://invalid.com/api", api_key="sk-test")

    def test_init_no_provider(self):
        """Test initialization without provider info."""
        with pytest.raises(ValueError, match="Must provide either provider"):
            ChatLimiter(api_key="sk-test")

    def test_init_state(self):
        """Test initial state setup."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        assert isinstance(limiter.state, LimiterState)
        assert limiter.state.request_limit is None  # No defaults - must be discovered
        assert limiter.state.token_limit is None  # No defaults - must be discovered
        assert limiter.state.requests_used == 0
        assert limiter.state.tokens_used == 0
        assert limiter.state.consecutive_rate_limit_errors == 0
        assert limiter._limits_discovered is False  # Limits not yet discovered
    
    def test_init_state_with_user_overrides(self):
        """Test initial state setup with user-provided limits."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, 
            api_key="sk-test",
            request_limit=1000,
            token_limit=50000
        )

        assert isinstance(limiter.state, LimiterState)
        assert limiter.state.request_limit == 1000  # User override
        assert limiter.state.token_limit == 50000  # User override
        assert limiter._limits_discovered is True  # User provided limits


class TestChatLimiterHeaders:
    """Tests for HTTP client header setup."""

    def test_openai_headers(self):
        """Test OpenAI headers setup."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        headers = limiter.async_client.headers
        assert headers["Authorization"] == "Bearer sk-test"
        assert "User-Agent" in headers
        assert "chat-limiter" in headers["User-Agent"]

    def test_anthropic_headers(self):
        """Test Anthropic headers setup."""
        limiter = ChatLimiter(provider=Provider.ANTHROPIC, api_key="sk-ant-test")

        headers = limiter.async_client.headers
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "User-Agent" in headers

    def test_openrouter_headers(self):
        """Test OpenRouter headers setup."""
        limiter = ChatLimiter(provider=Provider.OPENROUTER, api_key="sk-or-test")

        headers = limiter.async_client.headers
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert "HTTP-Referer" in headers
        assert "User-Agent" in headers

    def test_custom_headers(self):
        """Test custom headers are preserved."""
        custom_headers = {"Custom-Header": "custom-value"}
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", headers=custom_headers
        )

        headers = limiter.async_client.headers
        assert headers["Custom-Header"] == "custom-value"
        assert headers["Authorization"] == "Bearer sk-test"


class TestChatLimiterContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_async_client):
        """Test async context manager."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        async with limiter:
            assert limiter._async_context_active is True

        assert limiter._async_context_active is False
        mock_async_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_double_entry(self, mock_async_client):
        """Test async context manager double entry raises error."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        async with limiter:
            with pytest.raises(RuntimeError, match="already active"):
                async with limiter:
                    pass

    def test_sync_context_manager(self, mock_sync_client):
        """Test sync context manager."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test",
            sync_http_client=mock_sync_client,
        )

        with limiter:
            assert limiter._sync_context_active is True

        assert limiter._sync_context_active is False
        mock_sync_client.close.assert_called_once()

    def test_sync_context_manager_double_entry(self, mock_sync_client):
        """Test sync context manager double entry raises error."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test",
            sync_http_client=mock_sync_client,
        )

        with limiter:
            with pytest.raises(RuntimeError, match="already active"):
                with limiter:
                    pass


class TestChatLimiterRateLimitUpdates:
    """Tests for rate limit discovery and updates."""

    def test_update_rate_limits(self):
        """Test updating rate limits from response (discovery)."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        from chat_limiter.providers import RateLimitInfo

        # Initial state - no limits discovered yet
        assert limiter.state.request_limit is None
        assert limiter.state.token_limit is None
        assert limiter._limits_discovered is False

        # Discover limits from API response
        rate_limit_info = RateLimitInfo(
            requests_limit=1000,
            tokens_limit=60000,
            requests_remaining=999,
            tokens_remaining=59000,
        )

        limiter._update_rate_limits(rate_limit_info)

        # Limits should now be discovered and set
        assert limiter.state.request_limit == 1000
        assert limiter.state.token_limit == 60000
        assert limiter.state.last_rate_limit_info == rate_limit_info
        assert limiter._limits_discovered is True

    def test_update_rate_limits_incremental_discovery(self):
        """Test rate limit discovery when limits are discovered incrementally."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        
        from chat_limiter.providers import RateLimitInfo

        # Initial state - no limits discovered yet
        assert limiter.state.request_limit is None
        assert limiter.state.token_limit is None
        assert limiter._limits_discovered is False

        # First API response: Only request limit discovered
        rate_limit_info_1 = RateLimitInfo(
            requests_limit=1000,
            requests_remaining=999,
        )
        limiter._update_rate_limits(rate_limit_info_1)
        
        # Request limit should be set, but limits_discovered should still be False
        # because token limit is not yet discovered
        assert limiter.state.request_limit == 1000
        assert limiter.state.token_limit is None
        assert limiter._limits_discovered is False

        # Second API response: Token limit discovered
        rate_limit_info_2 = RateLimitInfo(
            requests_limit=1000,  # Same as before
            tokens_limit=60000,
            tokens_remaining=59000,
        )
        limiter._update_rate_limits(rate_limit_info_2)
        
        # Now both limits should be set and limits_discovered should be True
        assert limiter.state.request_limit == 1000
        assert limiter.state.token_limit == 60000
        assert limiter._limits_discovered is True

    def test_timeout_configuration(self):
        """Test that timeout parameter is properly configured."""
        # Test default timeout
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")
        assert limiter._user_timeout == 120.0
        assert limiter.async_client.timeout.read == 120.0
        assert limiter.sync_client.timeout.read == 120.0
        
        # Test custom timeout
        custom_timeout = 180.0
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test", timeout=custom_timeout)
        assert limiter._user_timeout == custom_timeout
        assert limiter.async_client.timeout.read == custom_timeout
        assert limiter.sync_client.timeout.read == custom_timeout

    def test_for_model_timeout_parameter(self):
        """Test that for_model method accepts and uses timeout parameter."""
        custom_timeout = 200.0
        limiter = ChatLimiter.for_model("gpt-4o", api_key="sk-test", timeout=custom_timeout)
        assert limiter._user_timeout == custom_timeout
        assert limiter.async_client.timeout.read == custom_timeout
        assert limiter.sync_client.timeout.read == custom_timeout

    @pytest.mark.asyncio
    async def test_enhanced_timeout_error_message(self):
        """Test that timeout errors include helpful information."""
        import httpx
        from unittest.mock import AsyncMock
        
        # Create a mock client that raises ReadTimeout
        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.ReadTimeout("Test timeout")
        
        limiter = ChatLimiter(
            provider=Provider.OPENAI, 
            api_key="sk-test", 
            timeout=90.0,
            http_client=mock_client
        )
        
        # Test that the enhanced error message is included
        with pytest.raises(httpx.ReadTimeout) as exc_info:
            async with limiter:
                await limiter.request("POST", "/test")
        
        error_message = str(exc_info.value)
        assert "💡 Timeout Error Help:" in error_message
        assert "Current timeout: 90.0s" in error_message
        assert "ChatLimiter.for_model('openai', timeout=150)" in error_message
        assert "reduce batch concurrency" in error_message

    def test_update_rate_limits_no_change(self):
        """Test rate limit update with no changes after initial discovery."""
        # Start with user-provided limits to have initialized limiters
        limiter = ChatLimiter(
            provider=Provider.OPENAI, 
            api_key="sk-test",
            request_limit=500,
            token_limit=30000
        )

        from chat_limiter.providers import RateLimitInfo

        # Store original limiters
        original_request_limiter = limiter.request_limiter
        original_token_limiter = limiter.token_limiter

        # Update with same limits (no change)
        rate_limit_info = RateLimitInfo(
            requests_limit=500,  # Same as current
            tokens_limit=30000,  # Same as current
        )

        limiter._update_rate_limits(rate_limit_info)

        # Limiters should not be recreated since limits didn't change
        assert limiter.request_limiter is original_request_limiter
        assert limiter.token_limiter is original_token_limiter

    def test_estimate_tokens(self):
        """Test token estimation."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        # Test with messages
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        estimated = limiter._estimate_tokens(request_data)
        assert estimated > 0
        assert estimated < 100  # Should be reasonable

        # Test with empty data
        estimated = limiter._estimate_tokens({})
        assert estimated == 0

        # Test with disabled estimation
        limiter.enable_token_estimation = False
        estimated = limiter._estimate_tokens(request_data)
        assert estimated == 0


class TestChatLimiterRequests:
    """Tests for making HTTP requests."""

    @pytest.mark.asyncio
    async def test_request_without_context(self, mock_async_client):
        """Test request without context manager raises error."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        with pytest.raises(
            RuntimeError, match="must be used as an async context manager"
        ):
            await limiter.request("GET", "/test")

    def test_request_sync_without_context(self, mock_sync_client):
        """Test sync request without context manager raises error."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test",
            sync_http_client=mock_sync_client,
        )

        with pytest.raises(
            RuntimeError, match="must be used as a sync context manager"
        ):
            limiter.request_sync("GET", "/test")

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_async_client, mock_openai_response):
        """Test successful request."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        # Mock the request
        mock_async_client.request.return_value = mock_openai_response

        async with limiter:
            response = await limiter.request(
                "POST", "/chat/completions", json={"test": "data"}
            )

        assert response == mock_openai_response
        mock_async_client.request.assert_called_once_with(
            "POST", "/chat/completions", json={"test": "data"}
        )

        # Check that rate limit info was updated
        assert limiter.state.last_rate_limit_info is not None
        assert limiter.state.requests_used == 1

    def test_successful_request_sync(self, mock_sync_client, mock_openai_response):
        """Test successful sync request."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test",
            sync_http_client=mock_sync_client,
        )

        # Mock the request
        mock_sync_client.request.return_value = mock_openai_response

        with limiter:
            response = limiter.request_sync(
                "POST", "/chat/completions", json={"test": "data"}
            )

        assert response == mock_openai_response
        mock_sync_client.request.assert_called_once_with(
            "POST", "/chat/completions", json={"test": "data"}
        )

        # Check that rate limit info was updated
        assert limiter.state.last_rate_limit_info is not None
        assert limiter.state.requests_used == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(
        self, mock_async_client, mock_rate_limit_response
    ):
        """Test handling of rate limit errors."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        # Mock the rate limit response
        mock_async_client.request.return_value = mock_rate_limit_response

        async with limiter:
            with patch("asyncio.sleep") as mock_sleep:
                # Import tenacity for the proper exception
                from tenacity import RetryError

                with pytest.raises(RetryError):  # tenacity wraps the original exception
                    await limiter.request("POST", "/chat/completions")

        # Should have slept according to retry-after header (called multiple times due to retries)
        assert mock_sleep.call_count >= 1
        # First call should be with retry-after value
        mock_sleep.assert_any_call(60.0)

        # Error count should be incremented (to max retries)
        assert limiter.state.consecutive_rate_limit_errors >= 1


class TestChatLimiterUtilities:
    """Tests for utility methods."""

    def test_get_current_limits(self):
        """Test getting current limits."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        limits = limiter.get_current_limits()

        assert limits["provider"] == "openai"
        assert limits["request_limit"] is None  # Not yet discovered
        assert limits["token_limit"] is None  # Not yet discovered
        assert limits["requests_used"] == 0
        assert limits["tokens_used"] == 0
        assert "last_request_time" in limits
        assert "last_limit_update" in limits
    
    def test_get_current_limits_with_user_overrides(self):
        """Test getting current limits with user-provided overrides."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, 
            api_key="sk-test",
            request_limit=1000,
            token_limit=50000
        )

        limits = limiter.get_current_limits()

        assert limits["provider"] == "openai"
        assert limits["request_limit"] == 1000  # User override
        assert limits["token_limit"] == 50000  # User override
        assert limits["requests_used"] == 0
        assert limits["tokens_used"] == 0

    def test_reset_usage_tracking(self):
        """Test resetting usage tracking."""
        limiter = ChatLimiter(provider=Provider.OPENAI, api_key="sk-test")

        # Set some usage
        limiter.state.requests_used = 10
        limiter.state.tokens_used = 1000
        limiter.state.consecutive_rate_limit_errors = 3

        # Reset
        limiter.reset_usage_tracking()

        assert limiter.state.requests_used == 0
        assert limiter.state.tokens_used == 0
        assert limiter.state.consecutive_rate_limit_errors == 0


class TestLimiterState:
    """Tests for LimiterState dataclass."""

    def test_limiter_state_defaults(self):
        """Test LimiterState default values."""
        state = LimiterState()

        assert state.request_limit is None  # No defaults - must be discovered
        assert state.token_limit is None  # No defaults - must be discovered
        assert state.requests_used == 0
        assert state.tokens_used == 0
        assert state.consecutive_rate_limit_errors == 0
        assert state.adaptive_backoff_factor == 1.0
        assert state.last_rate_limit_info is None
        assert isinstance(state.last_request_time, float)
        assert isinstance(state.last_limit_update, float)

    def test_limiter_state_custom_values(self):
        """Test LimiterState with custom values."""
        state = LimiterState(
            request_limit=100,
            token_limit=50000,
            requests_used=5,
            tokens_used=2500,
        )

        assert state.request_limit == 100
        assert state.token_limit == 50000
        assert state.requests_used == 5
        assert state.tokens_used == 2500
