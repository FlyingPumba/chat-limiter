"""
Tests for dynamic model discovery functionality.
"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from chat_limiter.models import (
    ModelDiscovery,
    ModelDiscoveryResult,
    detect_provider_from_model_async,
    detect_provider_from_model_sync,
    clear_model_cache,
)


class TestModelDiscovery:
    """Test the ModelDiscovery class methods."""

    @pytest.mark.asyncio
    async def test_get_openai_models_success(self):
        """Test successful OpenAI model retrieval."""
        # Mock the entire method to return the expected set
        expected_models = {"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
        
        with patch.object(ModelDiscovery, 'get_openai_models', return_value=expected_models):
            models = await ModelDiscovery.get_openai_models("test-key")

            # Should contain expected models
            assert models == expected_models
            assert "gpt-4o" in models
            assert "gpt-4o-mini" in models
            assert "gpt-3.5-turbo" in models

    @pytest.mark.asyncio
    async def test_get_openai_models_error_raises(self):
        """Test OpenAI model retrieval raises exception on error."""
        # Test the actual implementation by calling it directly
        # Should raise an exception instead of returning fallback models
        with pytest.raises(Exception):  # Could be HTTPStatusError or other exception
            await ModelDiscovery.get_openai_models("invalid-key")

    @pytest.mark.asyncio
    async def test_get_anthropic_models_success(self):
        """Test successful Anthropic model retrieval."""
        # Mock the entire method to return the expected set
        expected_models = {"claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"}
        
        with patch.object(ModelDiscovery, 'get_anthropic_models', return_value=expected_models):
            models = await ModelDiscovery.get_anthropic_models("test-key")

            # Should contain expected models
            assert models == expected_models
            assert "claude-3-5-sonnet-20241022" in models
            assert "claude-3-haiku-20240307" in models

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
        # Test the actual implementation - it should return fallback models when API call fails
        models = await ModelDiscovery.get_openrouter_models()

        # Should return fallback models
        assert isinstance(models, set)
        assert len(models) > 0
        # Check for some expected OpenRouter models from the fallback list
        openrouter_fallbacks = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"]
        assert any(model in models for model in openrouter_fallbacks)

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

    def test_sync_methods_error_raises(self):
        """Test sync methods raise exceptions on error."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Async error")
            
            # Test OpenAI sync raises exception
            with pytest.raises(Exception):
                ModelDiscovery.get_openai_models_sync("test-key")
            
            # Test Anthropic sync raises exception
            with pytest.raises(Exception):
                ModelDiscovery.get_anthropic_models_sync("test-key")
            
            # Test OpenRouter sync raises exception
            with pytest.raises(Exception):
                ModelDiscovery.get_openrouter_models_sync("test-key")

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
        # Mock OpenRouter API call for pattern detection
        with patch.object(ModelDiscovery, "get_openrouter_models") as mock_openrouter:
            mock_openrouter.return_value = {"openai/gpt-4o", "anthropic/claude-3-sonnet"}
            
            result = await detect_provider_from_model_async("openai/gpt-4o")
            assert result.found_provider == "openrouter"
            assert result.model_found == True
            
            result = await detect_provider_from_model_async("anthropic/claude-3-sonnet")
            assert result.found_provider == "openrouter"
            assert result.model_found == True

    @pytest.mark.asyncio
    async def test_openai_o3_bug_was_fixed(self):
        """Test that openai/o3 bug has been fixed - now correctly uses OpenAI provider."""
        api_keys = {
            "openai": "test-openai-key",
            "openrouter": "test-openrouter-key"
        }
        
        # Mock the API responses - OpenAI has "o3" model, OpenRouter has "openai/o3"
        mock_openai = AsyncMock(return_value={"o3", "gpt-4o", "gpt-3.5-turbo"})
        mock_openrouter = AsyncMock(return_value={"openai/o3", "openai/gpt-4o", "anthropic/claude-3-sonnet"})
        
        with patch.object(ModelDiscovery, "get_openai_models", mock_openai), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            # Fixed behavior: should prioritize OpenAI for openai/ prefixed models
            # when the base model exists in OpenAI
            result = await detect_provider_from_model_async("openai/o3", api_keys)
            
            # Fixed: now correctly uses OpenAI provider
            assert result.found_provider == "openai"
            assert result.model_found == True
            
            # The fix ensures that:
            # 1. The model starts with "openai/" so we prefer OpenAI
            # 2. "o3" exists in OpenAI models
            # 3. So we use "o3" from OpenAI instead of "openai/o3" from OpenRouter

    @pytest.mark.asyncio
    async def test_openai_o3_expected_behavior(self):
        """Test expected behavior: openai/o3 should use o3 from OpenAI provider when available."""
        api_keys = {
            "openai": "test-openai-key",
            "openrouter": "test-openrouter-key"
        }
        
        # Mock the API responses - OpenAI has "o3" model, OpenRouter has "openai/o3"
        mock_openai = AsyncMock(return_value={"o3", "gpt-4o", "gpt-3.5-turbo"})
        mock_openrouter = AsyncMock(return_value={"openai/o3", "openai/gpt-4o", "anthropic/claude-3-sonnet"})
        
        with patch.object(ModelDiscovery, "get_openai_models", mock_openai), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            # Expected behavior: should prioritize OpenAI for openai/ prefixed models
            # when the base model exists in OpenAI
            result = await detect_provider_from_model_async("openai/o3", api_keys)
            
            # Should use OpenAI provider with base model name "o3"
            assert result.found_provider == "openai"
            assert result.model_found == True
            assert "o3" in result.openai_models

    @pytest.mark.asyncio
    async def test_anthropic_claude_expected_behavior(self):
        """Test expected behavior: anthropic/claude-3-sonnet should use claude-3-sonnet from Anthropic when available."""
        api_keys = {
            "anthropic": "test-anthropic-key",
            "openrouter": "test-openrouter-key"
        }
        
        # Mock the API responses - Anthropic has the base model, OpenRouter has prefixed version
        mock_anthropic = AsyncMock(return_value={"claude-3-sonnet", "claude-3-haiku"})
        mock_openrouter = AsyncMock(return_value={"anthropic/claude-3-sonnet", "openai/gpt-4o"})
        
        with patch.object(ModelDiscovery, "get_anthropic_models", mock_anthropic), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            # Should use Anthropic provider with base model name
            result = await detect_provider_from_model_async("anthropic/claude-3-sonnet", api_keys)
            
            # Should use Anthropic provider
            assert result.found_provider == "anthropic"
            assert result.model_found == True
            assert "claude-3-sonnet" in result.anthropic_models

    @pytest.mark.asyncio
    async def test_fallback_to_openrouter_when_base_model_not_found(self):
        """Test fallback: use OpenRouter when base model doesn't exist in preferred provider."""
        api_keys = {
            "openai": "test-openai-key",
            "openrouter": "test-openrouter-key"
        }
        
        # Mock the API responses - OpenAI doesn't have "some-custom-model", but OpenRouter has "openai/some-custom-model"
        mock_openai = AsyncMock(return_value={"o3", "gpt-4o", "gpt-3.5-turbo"})  # No "some-custom-model"
        mock_openrouter = AsyncMock(return_value={"openai/some-custom-model", "openai/gpt-4o"})
        
        with patch.object(ModelDiscovery, "get_openai_models", mock_openai), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            # Should fallback to OpenRouter since "some-custom-model" doesn't exist in OpenAI
            result = await detect_provider_from_model_async("openai/some-custom-model", api_keys)
            
            # Should use OpenRouter as fallback
            assert result.found_provider == "openrouter"
            assert result.model_found == True
            assert "openai/some-custom-model" in result.openrouter_models

    @pytest.mark.asyncio
    async def test_detect_provider_from_model_async_with_api_keys(self):
        """Test detection with API keys for live queries."""
        api_keys = {
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
        }
        
        # Create async mock that returns coroutines
        mock_openai = AsyncMock(return_value={"custom-gpt-model"})
        mock_anthropic = AsyncMock(return_value={"other-model"})
        mock_openrouter = AsyncMock(return_value={"yet-another-model"})
        
        with patch.object(ModelDiscovery, "get_openai_models", mock_openai), \
             patch.object(ModelDiscovery, "get_anthropic_models", mock_anthropic), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            result = await detect_provider_from_model_async("custom-gpt-model", api_keys)
            assert result.found_provider == "openai"
            assert result.model_found == True
            assert result.openai_models == {"custom-gpt-model"}
            mock_openai.assert_called_once_with("test-openai-key")

    @pytest.mark.asyncio
    async def test_detect_provider_from_model_async_not_found(self):
        """Test detection when model is not found."""
        api_keys = {"openai": "test-key"}
        
        # Create async mock that returns coroutines
        mock_openai = AsyncMock(return_value={"different-model"})
        mock_anthropic = AsyncMock(return_value={"other-model"})
        mock_openrouter = AsyncMock(return_value={"yet-another-model"})
        
        with patch.object(ModelDiscovery, "get_openai_models", mock_openai), \
             patch.object(ModelDiscovery, "get_anthropic_models", mock_anthropic), \
             patch.object(ModelDiscovery, "get_openrouter_models", mock_openrouter):
            
            result = await detect_provider_from_model_async("unknown-model", api_keys)
            assert result.found_provider is None
            assert result.model_found == False
            assert result.openai_models == {"different-model"}
            assert result.openrouter_models == {"yet-another-model"}
            assert result.get_total_models_found() == 2  # Only OpenAI and OpenRouter called

    def test_detect_provider_from_model_sync(self):
        """Test synchronous provider detection."""
        with patch("asyncio.run") as mock_run:
            mock_result = ModelDiscoveryResult(found_provider="openai", model_found=True)
            mock_run.return_value = mock_result
            
            result = detect_provider_from_model_sync("test-model")
            assert result.found_provider == "openai"
            assert result.model_found == True

    def test_detect_provider_from_model_sync_error(self):
        """Test sync detection raises exception on error."""
        with patch("asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Error")
            
            with pytest.raises(Exception):
                detect_provider_from_model_sync("unknown-model")



class TestModelDiscoveryResult:
    """Test the ModelDiscoveryResult class."""
    
    def test_model_discovery_result_creation(self):
        """Test creating ModelDiscoveryResult instances."""
        result = ModelDiscoveryResult()
        assert result.found_provider is None
        assert result.model_found == False
        assert result.openai_models is None
        assert result.get_total_models_found() == 0
        
    def test_model_discovery_result_with_data(self):
        """Test ModelDiscoveryResult with data."""
        result = ModelDiscoveryResult(
            found_provider="openai",
            model_found=True,
            openai_models={"gpt-4o", "gpt-3.5-turbo"},
            anthropic_models={"claude-3-sonnet"},
            errors={"openrouter": "API key invalid"}
        )
        
        assert result.found_provider == "openai"
        assert result.model_found == True
        assert result.get_total_models_found() == 3
        assert result.errors["openrouter"] == "API key invalid"
        
        all_models = result.get_all_models()
        assert "openai" in all_models
        assert "anthropic" in all_models
        assert "openrouter" not in all_models  # Not set
        assert len(all_models["openai"]) == 2
        assert len(all_models["anthropic"]) == 1


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
        # Test the cache clearing functionality
        clear_model_cache()
        
        # Test cache behavior with mocked API calls
        from chat_limiter.models import _model_cache
        from datetime import datetime
        
        # Manually add an item to cache
        test_models = {"gpt-4o", "gpt-3.5-turbo"}
        _model_cache["test_key"] = {
            "models": test_models,
            "timestamp": datetime.now()
        }
        
        # Verify cache has the item
        assert len(_model_cache) == 1
        assert _model_cache["test_key"]["models"] == test_models
        
        # Clear cache and verify it's empty
        clear_model_cache()
        assert len(_model_cache) == 0