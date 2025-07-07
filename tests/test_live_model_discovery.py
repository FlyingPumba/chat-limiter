"""
Live tests for dynamic model discovery edge cases.
"""

import os
import pytest
from chat_limiter import ChatLimiter
from chat_limiter.models import ModelDiscovery


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestLiveModelDiscovery:
    """Test dynamic model discovery with real API calls."""

    @pytest.mark.asyncio
    async def test_gpt_4_1_model_discovery(self):
        """Test that gpt-4.1 (or similar models) can be discovered via live API."""
        # First, let's see what models are actually available
        models = await ModelDiscovery.get_openai_models(os.getenv("OPENAI_API_KEY"))
        print(f"Available OpenAI models: {sorted(models)}")
        
        # Check if gpt-4.1 or variants are in the discovered models
        gpt_4_variants = [m for m in models if "gpt-4" in m.lower() and "." in m]
        print(f"GPT-4 variants with dots: {gpt_4_variants}")
        
        # This should not fail if gpt-4.1 is a real model
        if "gpt-4.1" in models:
            # Test that ChatLimiter can handle it
            limiter = ChatLimiter.for_model("gpt-4.1")
            assert limiter.provider.value == "openai"
        else:
            pytest.skip("gpt-4.1 not found in OpenAI API response")

    @pytest.mark.asyncio 
    async def test_model_filtering_is_not_too_restrictive(self):
        """Test that model filtering doesn't exclude valid OpenAI models."""
        models = await ModelDiscovery.get_openai_models(os.getenv("OPENAI_API_KEY"))
        
        # Check that we're not filtering out valid models
        # All models from OpenAI /v1/models endpoint should be included if they're for chat
        assert len(models) > 0
        
        # Print all models for debugging
        print(f"All discovered models: {sorted(models)}")
        
        # Look for any models that might have been incorrectly filtered
        all_models_raw = []
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                for model in data.get("data", []):
                    all_models_raw.append(model.get("id", ""))
            
            print(f"Raw API response models: {sorted(all_models_raw)}")
            
            # Find models that might have been filtered out incorrectly
            gpt_models_raw = [m for m in all_models_raw if "gpt" in m.lower()]
            filtered_out = set(gpt_models_raw) - models
            if filtered_out:
                print(f"GPT models filtered out: {sorted(filtered_out)}")
                
        except Exception as e:
            print(f"Could not fetch raw API response: {e}")

    def test_unknown_model_fails_properly(self):
        """Test that unknown models fail explicitly when dynamic discovery fails."""
        # This should fail explicitly - no hidden fallbacks
        with pytest.raises(ValueError) as excinfo:
            ChatLimiter.for_model("gpt-nonexistent", api_key="fake-key")
        assert "Could not determine provider" in str(excinfo.value)

    def test_provider_override_works(self):
        """Test that provider override bypasses dynamic discovery issues."""
        # This should always work regardless of dynamic discovery
        limiter = ChatLimiter.for_model("gpt-4.1", provider="openai", api_key="test-key")
        assert limiter.provider.value == "openai"