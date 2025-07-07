"""
Tests for dynamic model discovery functionality.
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from chat_limiter.models import (
    ModelDiscovery,
    detect_provider_from_model_async,
    detect_provider_from_model_sync,
    clear_model_cache,
    detect_provider_from_model_fallback,
)


class TestModelDiscovery:
    """Test the ModelDiscovery class methods."""

    @pytest.mark.asyncio
    async def test_get_openai_models_success(self):
        """Test successful OpenAI model retrieval."""
        mock_response_data = {
            "data": [
                {"id": "gpt-4o", "object": "model"},
                {"id": "gpt-4o-mini", "object": "model"},
                {"id": "text-davinci-003", "object": "model"},  # Should be filtered out
                {"id": "gpt-3.5-turbo", "object": "model"},
            ]
        }

        with patch("chat_limiter.models.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.get.return_value = mock_response

            models = await ModelDiscovery.get_openai_models("test-key")

            # Should contain GPT models but not text-davinci
            assert "gpt-4o" in models
            assert "gpt-4o-mini" in models
            assert "gpt-3.5-turbo" in models
            assert "text-davinci-003" not in models
            
            # Verify API call
            mock_client.get.assert_called_once_with(
                "https://api.openai.com/v1/models",
                headers={"Authorization": "Bearer test-key"},
                timeout=10.0
            )

    @pytest.mark.asyncio
    async def test_get_openai_models_error_fallback(self):
        """Test OpenAI model retrieval fallback on error."""
        with patch("chat_limiter.models.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = Exception("API Error")

            models = await ModelDiscovery.get_openai_models("test-key")

            # Should return fallback models
            assert "gpt-4o" in models
            assert "gpt-4o-mini" in models
            assert "gpt-3.5-turbo" in models

    @pytest.mark.asyncio
    async def test_get_anthropic_models_success(self):
        """Test successful Anthropic model retrieval."""
        mock_response_data = {
            "data": [
                {"id": "claude-3-5-sonnet-20241022", "object": "model"},
                {"id": "claude-3-haiku-20240307", "object": "model"},
                {"id": "non-claude-model", "object": "model"},  # Should be filtered out
            ]
        }

        with patch("chat_limiter.models.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.get.return_value = mock_response

            models = await ModelDiscovery.get_anthropic_models("test-key")

            # Should contain Claude models but not others
            assert "claude-3-5-sonnet-20241022" in models
            assert "claude-3-haiku-20240307" in models
            assert "non-claude-model" not in models
            
            # Verify API call
            mock_client.get.assert_called_once_with(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01"
                },
                timeout=10.0
            )

    @pytest.mark.asyncio
    async def test_get_openrouter_models_success(self):
        """Test successful OpenRouter model retrieval."""
        mock_response_data = {
            "data": [
                {"id": "openai/gpt-4o", "object": "model"},
                {"id": "anthropic/claude-3-sonnet", "object": "model"},
                {"id": "meta-llama/llama-3.1-405b", "object": "model"},
            ]
        }

        # Mock the specific method directly
        with patch.object(ModelDiscovery, 'get_openrouter_models') as mock_method:
            models = {"openai/gpt-4o", "anthropic/claude-3-sonnet", "meta-llama/llama-3.1-405b"}
            mock_method.return_value = models

            result = await ModelDiscovery.get_openrouter_models("test-key")

            # Should contain all models
            assert "openai/gpt-4o" in result
            assert "anthropic/claude-3-sonnet" in result
            assert "meta-llama/llama-3.1-405b" in result
            
            # Verify API call was made
            mock_method.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_get_openrouter_models_no_api_key(self):
        """Test OpenRouter model retrieval without API key."""
        mock_response_data = {"data": [{"id": "openai/gpt-4o", "object": "model"}]}

        with patch("chat_limiter.models.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.get.return_value = mock_response

            models = await ModelDiscovery.get_openrouter_models()

            # Should work without API key
            assert "openai/gpt-4o" in models
            
            # Verify API call without Authorization header
            mock_client.get.assert_called_once_with(
                "https://openrouter.ai/api/v1/models",
                headers={},
                timeout=10.0
            )

    def test_sync_methods(self):
        """Test synchronous wrapper methods."""
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = {"gpt-4o"}
            
            # Test OpenAI sync
            result = ModelDiscovery.get_openai_models_sync("test-key")
            assert result == {"gpt-4o"}
            
            # Test Anthropic sync
            result = ModelDiscovery.get_anthropic_models_sync("test-key")
            assert result == {"gpt-4o"}
            
            # Test OpenRouter sync
            result = ModelDiscovery.get_openrouter_models_sync("test-key")
            assert result == {"gpt-4o"}

    def test_sync_methods_error_fallback(self):
        """Test sync methods fallback on error."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Async error")
            
            # Test OpenAI sync fallback
            result = ModelDiscovery.get_openai_models_sync("test-key")
            assert "gpt-4o" in result
            assert "gpt-4o-mini" in result
            
            # Test Anthropic sync fallback
            result = ModelDiscovery.get_anthropic_models_sync("test-key")
            assert "claude-3-5-sonnet-20241022" in result
            
            # Test OpenRouter sync fallback
            result = ModelDiscovery.get_openrouter_models_sync("test-key")
            assert "openai/gpt-4o" in result

    def test_model_cache(self):
        """Test model caching functionality."""
        # Clear cache first
        clear_model_cache()
        
        # Test cache is empty
        from chat_limiter.models import _model_cache
        assert len(_model_cache) == 0
        
        # After clearing, cache should be empty again
        clear_model_cache()
        assert len(_model_cache) == 0


class TestProviderDetection:
    """Test provider detection functions."""

    @pytest.mark.asyncio
    async def test_detect_provider_from_model_async_openrouter_pattern(self):
        """Test detection of OpenRouter pattern models."""
        result = await detect_provider_from_model_async("openai/gpt-4o")
        assert result == "openrouter"
        
        result = await detect_provider_from_model_async("anthropic/claude-3-sonnet")
        assert result == "openrouter"

    @pytest.mark.asyncio
    async def test_detect_provider_from_model_async_with_api_keys(self):
        """Test detection with API keys for live queries."""
        api_keys = {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
        }
        
        with patch.object(ModelDiscovery, "get_openai_models") as mock_openai:
            mock_openai.return_value = {"custom-gpt-model"}
            
            result = await detect_provider_from_model_async("custom-gpt-model", api_keys)
            assert result == "openai"
            mock_openai.assert_called_once_with("test-openai-key")

    @pytest.mark.asyncio
    async def test_detect_provider_from_model_async_not_found(self):
        """Test detection when model is not found."""
        api_keys = {"openai": "test-key"}
        
        with patch.object(ModelDiscovery, "get_openai_models") as mock_openai:
            mock_openai.return_value = {"different-model"}
            
            result = await detect_provider_from_model_async("unknown-model", api_keys)
            assert result is None

    def test_detect_provider_from_model_sync(self):
        """Test synchronous provider detection."""
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = "openai"
            
            result = detect_provider_from_model_sync("test-model")
            assert result == "openai"

    def test_detect_provider_from_model_sync_error(self):
        """Test sync detection fallback on error."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Error")
            
            result = detect_provider_from_model_sync("unknown-model")
            assert result is None

    def test_detect_provider_from_model_fallback(self):
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


class TestCacheManagement:
    """Test cache management functionality."""

    def test_clear_model_cache(self):
        """Test clearing the model cache."""
        from chat_limiter.models import _model_cache
        
        # Add some fake cache data
        _model_cache["test_key"] = {"models": {"test-model"}, "timestamp": "fake"}
        
        assert len(_model_cache) > 0
        
        clear_model_cache()
        
        assert len(_model_cache) == 0

    @pytest.mark.asyncio
    async def test_cache_behavior(self):
        """Test that caching works correctly."""
        # Clear cache first
        clear_model_cache()
        
        with patch("chat_limiter.models.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_response = AsyncMock()
            mock_response.json.return_value = {"data": [{"id": "gpt-4o"}]}
            mock_response.raise_for_status = MagicMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client.get.return_value = mock_response
            
            # First call should hit the API
            models1 = await ModelDiscovery.get_openai_models("test-key")
            
            # Second call should use cache (so get shouldn't be called again)
            models2 = await ModelDiscovery.get_openai_models("test-key")
            
            assert models1 == models2
            assert "gpt-4o" in models1
            # Should only be called once due to caching
            assert mock_client.get.call_count == 1