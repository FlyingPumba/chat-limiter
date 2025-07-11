"""
Live test for o3-mini with temperature restrictions.

This test demonstrates the temperature handling issue with reasoning models
and verifies that the OpenAI adapter properly handles temperature restrictions.
"""

import os
import pytest
from unittest.mock import patch

from chat_limiter import ChatLimiter, Provider
from chat_limiter.types import ChatCompletionRequest, Message, MessageRole


class TestO3MiniTemperature:
    """Test temperature handling for o3-mini reasoning model."""

    @pytest.fixture
    def openai_api_key(self):
        """Get OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return api_key

    @pytest.mark.asyncio
    async def test_o3_mini_with_invalid_temperature(self, openai_api_key):
        """Test o3-mini with temperature=1e-19 (should be handled gracefully)."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key=openai_api_key,
            timeout=30.0
        )

        async with limiter:
            # This should work after we implement the temperature handling
            response = await limiter.chat_completion(
                model="o3-mini",
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                max_tokens=200,  # Increased for reasoning models
                temperature=1e-19  # This should cause issues with current implementation
            )
            
            # Should not have error after we implement the fix
            assert response.success
            assert response.choices
            # Check that we got a response (content might be empty due to finish_reason='length')
            assert len(response.choices) > 0
            assert response.choices[0].message.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_o3_mini_with_no_temperature(self, openai_api_key):
        """Test o3-mini with no temperature specified (should default to 1)."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key=openai_api_key,
            timeout=30.0
        )

        async with limiter:
            response = await limiter.chat_completion(
                model="o3-mini",
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                max_tokens=200  # Increased for reasoning models
                # No temperature specified - should default to 1 for reasoning models
            )
            
            assert response.success
            assert response.choices
            assert len(response.choices) > 0
            assert response.choices[0].message.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_o3_mini_with_valid_temperature(self, openai_api_key):
        """Test o3-mini with temperature=1 (should work fine)."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key=openai_api_key,
            timeout=30.0
        )

        async with limiter:
            response = await limiter.chat_completion(
                model="o3-mini",
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                max_tokens=200,  # Increased for reasoning models
                temperature=1.0  # Should work fine for reasoning models
            )
            
            assert response.success
            assert response.choices
            assert len(response.choices) > 0
            assert response.choices[0].message.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_o3_mini_temperature_warning(self, openai_api_key, capsys):
        """Test that a warning is printed when user provides different temperature for reasoning model."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key=openai_api_key,
            timeout=30.0
        )

        async with limiter:
            response = await limiter.chat_completion(
                model="o3-mini",
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                max_tokens=200,  # Increased for reasoning models
                temperature=0.5  # Different from default temperature=1
            )
            
            # Should still work but with warning
            assert response.success
            assert response.choices
            assert len(response.choices) > 0
            assert response.choices[0].message.role == MessageRole.ASSISTANT
            
            # Check that warning was printed
            captured = capsys.readouterr()
            assert "WARNING" in captured.out or "WARNING" in captured.err

    @pytest.mark.asyncio
    async def test_non_reasoning_model_temperature(self, openai_api_key):
        """Test that non-reasoning models still work with low temperature."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key=openai_api_key,
            timeout=30.0
        )

        async with limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                max_tokens=100,  # Sufficient for non-reasoning models
                temperature=1e-19  # Should work fine for non-reasoning models
            )
            
            assert response.success
            assert response.choices
            assert len(response.choices) > 0
            assert response.choices[0].message.role == MessageRole.ASSISTANT
            # GPT-4o should actually return content
            assert response.choices[0].message.content.strip()