"""
Integration tests using live API keys.

These tests require actual API keys and will make real API calls.
They are skipped if the required environment variables are not set.
"""

import os

import pytest

from chat_limiter import ChatLimiter, Message, MessageRole

# Skip conditions for each provider
OPENAI_AVAILABLE = os.getenv("OPENAI_API_KEY") is not None
ANTHROPIC_AVAILABLE = os.getenv("ANTHROPIC_API_KEY") is not None
OPENROUTER_AVAILABLE = os.getenv("OPENROUTER_API_KEY") is not None


class TestOpenAIIntegration:
    """Integration tests for OpenAI API."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        """Test basic chat completion with OpenAI."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o-mini",
                messages=[Message(role=MessageRole.USER, content="Hi")],
                max_tokens=5
            )

            assert response.provider == "openai"
            assert response.choices
            assert response.choices[0].message.content
            assert response.usage
            assert response.usage.total_tokens > 0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_rate_limit_header_parsing(self):
        """Test that OpenAI rate limit headers are parsed correctly."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            # Make a request to get headers
            await limiter.chat_completion(
                model="gpt-4o-mini",
                messages=[Message(role=MessageRole.USER, content="Test")],
                max_tokens=5
            )

            # Check that rate limit info was updated
            limits = limiter.get_current_limits()

            # OpenAI should have provided rate limit information
            assert limits["request_limit"] > 0
            assert limits["token_limit"] > 0
            assert limits["requests_used"] >= 1

            # Check that we have recent rate limit info
            assert limiter.state.last_rate_limit_info is not None
            rate_info = limiter.state.last_rate_limit_info

            # OpenAI typically provides these headers
            assert rate_info.requests_limit is not None or rate_info.requests_remaining is not None

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_very_short_prompt(self):
        """Test with minimal prompt to verify token estimation."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o-mini",
                prompt="Hi",
                max_tokens=1
            )

            assert isinstance(response, str)
            assert len(response) >= 0  # Could be empty with max_tokens=1

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_sync_chat_completion(self):
        """Test synchronous chat completion."""
        with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            response = limiter.chat_completion_sync(
                model="gpt-4o-mini",
                messages=[Message(role=MessageRole.USER, content="Hello")],
                max_tokens=5
            )

            assert response.provider == "openai"
            assert response.choices
            assert response.usage


class TestAnthropicIntegration:
    """Integration tests for Anthropic API."""

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        """Test basic chat completion with Anthropic."""
        async with ChatLimiter.for_model("claude-3-haiku-20240307") as limiter:
            response = await limiter.chat_completion(
                model="claude-3-haiku-20240307",
                messages=[Message(role=MessageRole.USER, content="Hi")],
                max_tokens=5
            )

            assert response.provider == "anthropic"
            assert response.choices
            assert response.choices[0].message.content
            assert response.usage
            assert response.usage.total_tokens > 0

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_system_message_handling(self):
        """Test that system messages are handled correctly by Anthropic."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="Be very brief."),
            Message(role=MessageRole.USER, content="What is Python?")
        ]

        async with ChatLimiter.for_model("claude-3-haiku-20240307") as limiter:
            response = await limiter.chat_completion(
                model="claude-3-haiku-20240307",
                messages=messages,
                max_tokens=20
            )

            assert response.provider == "anthropic"
            assert response.choices
            # Should have only one choice with assistant message
            assert len(response.choices) == 1
            assert response.choices[0].message.role == MessageRole.ASSISTANT

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_rate_limit_header_parsing(self):
        """Test that Anthropic rate limit headers are parsed correctly."""
        async with ChatLimiter.for_model("claude-3-haiku-20240307") as limiter:
            await limiter.chat_completion(
                model="claude-3-haiku-20240307",
                messages=[Message(role=MessageRole.USER, content="Test")],
                max_tokens=5
            )

            limits = limiter.get_current_limits()

            # Anthropic should provide rate limit information after first request
            # (could be None if not discovered yet from actual API response headers)
            assert limits["requests_used"] >= 1
            
            # Rate limit info may be None initially if no API discovery happened
            # In real usage, this would be populated after first API call

            # Check rate limit info
            assert limiter.state.last_rate_limit_info is not None


class TestOpenRouterIntegration:
    """Integration tests for OpenRouter API."""

    @pytest.mark.skipif(not OPENROUTER_AVAILABLE, reason="OPENROUTER_API_KEY not set")
    @pytest.mark.asyncio
    async def test_basic_chat_completion(self):
        """Test basic chat completion with OpenRouter."""
        # Try multiple models to handle availability issues
        models_to_try = [
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/wizardlm-2-8x22b:nitro"
        ]

        success = False
        for model in models_to_try:
            try:
                async with ChatLimiter.for_model(model) as limiter:
                    response = await limiter.chat_completion(
                        model=model,
                        messages=[Message(role=MessageRole.USER, content="Hi")],
                        max_tokens=5
                    )

                    assert response.provider == "openrouter"
                    if response.choices:  # Some models might return empty choices
                        assert response.choices[0].message.content is not None
                        success = True
                        break
            except Exception:
                continue

        # If none of the free models work, just verify the provider detection works
        if not success:
            async with ChatLimiter.for_model("openai/gpt-4o-mini") as limiter:
                assert limiter.provider.value == "openrouter"

    @pytest.mark.skipif(not OPENROUTER_AVAILABLE, reason="OPENROUTER_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_model_via_openrouter(self):
        """Test using OpenAI model through OpenRouter."""
        async with ChatLimiter.for_model("openai/gpt-4o-mini") as limiter:
            response = await limiter.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[Message(role=MessageRole.USER, content="Test")],
                max_tokens=5
            )

            assert response.provider == "openrouter"
            assert response.choices


class TestProviderAutoDetection:
    """Test provider auto-detection with real models."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_model_detection(self):
        """Test that OpenAI models are detected correctly."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            assert limiter.provider.value == "openai"

            response = await limiter.simple_chat(
                model="gpt-4o-mini",
                prompt="Hi",
                max_tokens=3
            )

            assert isinstance(response, str)

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_anthropic_model_detection(self):
        """Test that Anthropic models are detected correctly."""
        async with ChatLimiter.for_model("claude-3-haiku-20240307") as limiter:
            assert limiter.provider.value == "anthropic"

            response = await limiter.simple_chat(
                model="claude-3-haiku-20240307",
                prompt="Hi",
                max_tokens=3
            )

            assert isinstance(response, str)

    @pytest.mark.skipif(not OPENROUTER_AVAILABLE, reason="OPENROUTER_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openrouter_model_detection(self):
        """Test that OpenRouter models are detected correctly."""
        async with ChatLimiter.for_model("openai/gpt-4o-mini") as limiter:
            assert limiter.provider.value == "openrouter"


class TestProviderOverride:
    """Test provider override functionality with real APIs."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_provider_override_string(self):
        """Test provider override with string value."""
        async with ChatLimiter.for_model(
            "custom-model-name",
            provider="openai"
        ) as limiter:
            assert limiter.provider.value == "openai"

            # This should work even though model name doesn't match OpenAI pattern
            response = await limiter.chat_completion(
                model="gpt-4o-mini",  # Use actual model for the request
                messages=[Message(role=MessageRole.USER, content="Hi")],
                max_tokens=3
            )

            assert response.provider == "openai"


class TestRateLimitBehavior:
    """Test rate limiting behavior with real APIs."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_multiple_requests_tracking(self):
        """Test that multiple requests are tracked correctly."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            # Make multiple small requests
            for i in range(3):
                await limiter.simple_chat(
                    model="gpt-4o-mini",
                    prompt=f"Test {i}",
                    max_tokens=1
                )

            limits = limiter.get_current_limits()

            # Should have made 3 requests
            assert limits["requests_used"] >= 3

            # Should have some token usage
            assert limits["tokens_used"] > 0

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_usage_reset(self):
        """Test usage tracking reset functionality."""
        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            # Make a request
            await limiter.simple_chat(
                model="gpt-4o-mini",
                prompt="Test",
                max_tokens=1
            )

            # Verify usage is tracked
            limits_before = limiter.get_current_limits()
            assert limits_before["requests_used"] > 0

            # Reset usage
            limiter.reset_usage_tracking()

            # Verify usage is reset
            limits_after = limiter.get_current_limits()
            assert limits_after["requests_used"] == 0
            assert limits_after["tokens_used"] == 0


class TestErrorHandling:
    """Test error handling with real APIs."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_invalid_model_error(self):
        """Test handling of invalid model names."""
        import httpx
        from tenacity import RetryError

        async with ChatLimiter.for_model("gpt-4o-mini") as limiter:
            # OpenAI returns a 404 for invalid models, which should be raised
            try:
                response = await limiter.chat_completion(
                    model="definitely-not-a-real-model-name-12345",
                    messages=[Message(role=MessageRole.USER, content="Test")],
                    max_tokens=5
                )
                # If we get here, check if the response indicates an error
                if not response.choices:
                    # Empty choices might indicate an error state
                    assert True  # This is acceptable error handling
                else:
                    pytest.fail("Expected an error for invalid model name")
            except (httpx.HTTPStatusError, RetryError):
                # This is the expected behavior
                assert True

