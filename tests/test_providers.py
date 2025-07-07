"""Tests for provider configurations and rate limit parsing."""

import pytest

from chat_limiter.providers import (
    Provider,
    ProviderConfig,
    RateLimitInfo,
    detect_provider_from_url,
    extract_rate_limit_info,
    get_provider_config,
)


class TestProvider:
    """Tests for Provider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENROUTER.value == "openrouter"

    def test_provider_count(self):
        """Test that we have expected number of providers."""
        assert len(Provider) == 3


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_provider_config_creation(self):
        """Test creating a provider config."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            base_url="https://api.openai.com/v1",
            default_request_limit=500,
            default_token_limit=30000,
        )

        assert config.provider == Provider.OPENAI
        assert config.base_url == "https://api.openai.com/v1"
        assert config.default_request_limit == 500
        assert config.default_token_limit == 30000
        assert config.request_buffer_ratio == 0.9  # Default value

    def test_provider_config_validation(self):
        """Test provider config validation."""
        # No validation is done on ProviderConfig creation anymore
        # since defaults can be None and must be discovered from API
        config = ProviderConfig(
            provider=Provider.OPENAI,
            base_url="https://api.openai.com/v1",
            default_request_limit=None,  # No defaults - must be discovered
        )
        
        # Config should be valid even with None defaults
        assert config.provider == Provider.OPENAI
        assert config.default_request_limit is None
        
        # Buffer ratio validation should still work
        with pytest.raises(ValueError):
            ProviderConfig(
                provider=Provider.OPENAI,
                base_url="https://api.openai.com/v1",
                request_buffer_ratio=0.05,  # Should be >= 0.1
            )

        with pytest.raises(ValueError):
            ProviderConfig(
                provider=Provider.OPENAI,
                base_url="https://api.openai.com/v1",
                request_buffer_ratio=1.5,  # Should be <= 1.0
            )


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_get_openai_config(self):
        """Test getting OpenAI config."""
        config = get_provider_config(Provider.OPENAI)

        assert config.provider == Provider.OPENAI
        assert config.base_url == "https://api.openai.com/v1"
        assert config.request_limit_header == "x-ratelimit-limit-requests"
        assert config.request_remaining_header == "x-ratelimit-remaining-requests"
        assert config.token_limit_header == "x-ratelimit-limit-tokens"
        assert config.default_request_limit is None  # No defaults - must be discovered
        assert config.default_token_limit is None  # No defaults - must be discovered

    def test_get_anthropic_config(self):
        """Test getting Anthropic config."""
        config = get_provider_config(Provider.ANTHROPIC)

        assert config.provider == Provider.ANTHROPIC
        assert config.base_url == "https://api.anthropic.com/v1"
        assert (
            config.request_remaining_header == "anthropic-ratelimit-requests-remaining"
        )
        assert config.token_limit_header == "anthropic-ratelimit-tokens-limit"
        assert config.default_request_limit is None  # No defaults - must be discovered
        assert config.default_token_limit is None  # No defaults - must be discovered

    def test_get_openrouter_config(self):
        """Test getting OpenRouter config."""
        config = get_provider_config(Provider.OPENROUTER)

        assert config.provider == Provider.OPENROUTER
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.auth_endpoint == "https://openrouter.ai/api/v1/auth/key"
        assert config.default_request_limit is None  # No defaults - must be discovered
        assert config.default_token_limit is None  # No defaults - must be discovered


class TestDetectProviderFromUrl:
    """Tests for detect_provider_from_url function."""

    def test_detect_openai(self):
        """Test detecting OpenAI from URL."""
        assert (
            detect_provider_from_url("https://api.openai.com/v1/chat/completions")
            == Provider.OPENAI
        )
        assert detect_provider_from_url("https://api.openai.com/v1") == Provider.OPENAI
        assert detect_provider_from_url("http://api.openai.com/v1") == Provider.OPENAI

    def test_detect_anthropic(self):
        """Test detecting Anthropic from URL."""
        assert (
            detect_provider_from_url("https://api.anthropic.com/v1/messages")
            == Provider.ANTHROPIC
        )
        assert (
            detect_provider_from_url("https://api.anthropic.com/v1")
            == Provider.ANTHROPIC
        )
        assert (
            detect_provider_from_url("http://api.anthropic.com/v1")
            == Provider.ANTHROPIC
        )

    def test_detect_openrouter(self):
        """Test detecting OpenRouter from URL."""
        assert (
            detect_provider_from_url("https://openrouter.ai/api/v1/chat/completions")
            == Provider.OPENROUTER
        )
        assert (
            detect_provider_from_url("https://openrouter.ai/api/v1")
            == Provider.OPENROUTER
        )
        assert (
            detect_provider_from_url("http://openrouter.ai/api/v1")
            == Provider.OPENROUTER
        )

    def test_detect_unknown(self):
        """Test detecting unknown provider."""
        assert detect_provider_from_url("https://api.unknown.com/v1") is None
        assert detect_provider_from_url("https://example.com/api") is None
        assert detect_provider_from_url("invalid-url") is None


class TestExtractRateLimitInfo:
    """Tests for extract_rate_limit_info function."""

    def test_extract_openai_headers(self):
        """Test extracting OpenAI rate limit headers."""
        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-limit-tokens": "30000",
            "x-ratelimit-remaining-tokens": "29500",
            "x-ratelimit-reset-tokens": "60",
            "content-type": "application/json",
        }

        config = get_provider_config(Provider.OPENAI)
        info = extract_rate_limit_info(headers, config)

        assert info.requests_limit == 500
        assert info.requests_remaining == 499
        assert info.requests_reset == 60
        assert info.tokens_limit == 30000
        assert info.tokens_remaining == 29500
        assert info.tokens_reset == 60
        assert info.metadata == headers

    def test_extract_anthropic_headers(self):
        """Test extracting Anthropic rate limit headers."""
        headers = {
            "anthropic-ratelimit-requests-remaining": "59",
            "anthropic-ratelimit-tokens-limit": "1000000",
            "anthropic-ratelimit-tokens-reset": "60",
            "retry-after": "30",
            "content-type": "application/json",
        }

        config = get_provider_config(Provider.ANTHROPIC)
        info = extract_rate_limit_info(headers, config)

        assert info.requests_remaining == 59
        assert info.tokens_limit == 1000000
        assert info.tokens_reset == 60
        assert info.retry_after == 30
        assert info.metadata == headers

    def test_extract_missing_headers(self):
        """Test extracting with missing headers."""
        headers = {
            "content-type": "application/json",
        }

        config = get_provider_config(Provider.OPENAI)
        info = extract_rate_limit_info(headers, config)

        assert info.requests_limit is None
        assert info.requests_remaining is None
        assert info.tokens_limit is None
        assert info.tokens_remaining is None
        assert info.retry_after is None
        assert info.metadata == headers

    def test_extract_invalid_headers(self):
        """Test extracting with invalid header values."""
        headers = {
            "x-ratelimit-limit-requests": "invalid",
            "x-ratelimit-remaining-requests": "not-a-number",
            "x-ratelimit-reset-requests": "abc",
            "retry-after": "xyz",
        }

        config = get_provider_config(Provider.OPENAI)
        info = extract_rate_limit_info(headers, config)

        assert info.requests_limit is None
        assert info.requests_remaining is None
        assert info.requests_reset is None
        assert info.retry_after is None
        assert info.metadata == headers


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_rate_limit_info_creation(self):
        """Test creating RateLimitInfo."""
        info = RateLimitInfo(
            requests_limit=500,
            requests_remaining=499,
            tokens_limit=30000,
            tokens_remaining=29500,
            retry_after=60.0,
        )

        assert info.requests_limit == 500
        assert info.requests_remaining == 499
        assert info.tokens_limit == 30000
        assert info.tokens_remaining == 29500
        assert info.retry_after == 60.0
        assert info.metadata == {}

    def test_rate_limit_info_defaults(self):
        """Test RateLimitInfo with default values."""
        info = RateLimitInfo()

        assert info.requests_limit is None
        assert info.requests_remaining is None
        assert info.tokens_limit is None
        assert info.tokens_remaining is None
        assert info.retry_after is None
        assert info.metadata == {}
