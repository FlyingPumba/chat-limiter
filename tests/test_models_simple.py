"""
Simple tests for dynamic model discovery functionality.
"""

import pytest
from chat_limiter.models import detect_provider_from_model_fallback, clear_model_cache
from chat_limiter.types import detect_provider_from_model


class TestBasicFunctionality:
    """Test basic functionality without complex mocking."""

    def test_fallback_provider_detection(self):
        """Test fallback provider detection with hardcoded lists."""
        # Test OpenAI models
        assert detect_provider_from_model_fallback("gpt-4o") == "openai"
        assert detect_provider_from_model_fallback("gpt-3.5-turbo") == "openai"
        
        # Test Anthropic models
        assert detect_provider_from_model_fallback("claude-3-5-sonnet-20241022") == "anthropic"
        assert detect_provider_from_model_fallback("claude-3-haiku-20240307") == "anthropic"
        
        # Test OpenRouter pattern
        assert detect_provider_from_model_fallback("openai/gpt-4o") == "openrouter"
        assert detect_provider_from_model_fallback("anthropic/claude-3-sonnet") == "openrouter"
        
        # Test unknown model
        assert detect_provider_from_model_fallback("unknown-model") is None

    def test_types_detect_provider_basic(self):
        """Test the main detect_provider_from_model function without dynamic discovery."""
        # Test known models (should use hardcoded lists)
        assert detect_provider_from_model("gpt-4o") == "openai"
        assert detect_provider_from_model("claude-3-haiku-20240307") == "anthropic"
        assert detect_provider_from_model("openai/gpt-4o") == "openrouter"
        
        # Test unknown model without dynamic discovery
        assert detect_provider_from_model("unknown-model") is None

    def test_cache_management(self):
        """Test cache clearing functionality."""
        # This should work without issues
        clear_model_cache()
        assert True  # If we get here, the function works

    def test_dynamic_discovery_disabled(self):
        """Test behavior when dynamic discovery is explicitly disabled."""
        # Should fall back to hardcoded detection
        result = detect_provider_from_model("gpt-4o", use_dynamic_discovery=False)
        assert result == "openai"
        
        # Unknown model should return None
        result = detect_provider_from_model("unknown-model", use_dynamic_discovery=False)
        assert result is None

    def test_dynamic_discovery_no_api_keys(self):
        """Test behavior when dynamic discovery is enabled but no API keys provided."""
        # Should fall back to hardcoded detection
        result = detect_provider_from_model("gpt-4o", use_dynamic_discovery=True, api_keys={})
        assert result == "openai"
        
        # Unknown model should return None
        result = detect_provider_from_model("unknown-model", use_dynamic_discovery=True, api_keys={})
        assert result is None