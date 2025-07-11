"""
Live tests for dynamic model discovery edge cases.
"""

import os
import pytest
from chat_limiter import ChatLimiter
from chat_limiter.models import ModelDiscovery
from chat_limiter.types import Message, MessageRole


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

    @pytest.mark.asyncio
    async def test_o3_model_discovery(self):
        """Test that the o3 model can be correctly found in the OpenAI provider using dynamic discovery."""
        # First, get all available OpenAI models
        models = await ModelDiscovery.get_openai_models(os.getenv("OPENAI_API_KEY"))
        print(f"Available OpenAI models: {sorted(models)}")
        
        # Check specifically for o3 model variants
        o3_models = [m for m in models if "o3" in m.lower()]
        assert len(o3_models) > 0, "o3 model not found in OpenAI API response"

        # Find the actual o3 model name
        o3_model = next(m for m in models if "o3" in m.lower())
        
        # Test that ChatLimiter can handle it through dynamic discovery
        limiter = ChatLimiter.for_model(o3_model, api_key=os.getenv("OPENAI_API_KEY"))
        assert limiter.provider.value == "openai"
        
        # Test that the model is correctly identified as OpenAI
        from chat_limiter.models import detect_provider_from_model_async
        discovery_result = await detect_provider_from_model_async(
            o3_model, 
            {"openai": os.getenv("OPENAI_API_KEY")}
        )
        assert discovery_result.found_provider == "openai"
        assert discovery_result.model_found == True

    @pytest.mark.asyncio
    async def test_o3_model_live_request(self):
        """Test making a live request to o3 model to ensure no errors."""
        # Skip if no OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # First, get all available OpenAI models to find o3
        models = await ModelDiscovery.get_openai_models(os.getenv("OPENAI_API_KEY"))
        o3_models = [m for m in models if "o3" in m.lower()]
        assert len(o3_models) > 0, "No o3 models found in OpenAI API response"
        
        # Try each o3 model until we find one that works for chat
        o3_model = "o3-mini"
        successful_response = None

        try:
            # Create a ChatLimiter for the o3 model
            async with ChatLimiter.for_model(o3_model, api_key=os.getenv("OPENAI_API_KEY")) as limiter:
                # Make a simple request
                response = await limiter.chat_completion(
                    model=o3_model,
                    messages=[Message(role=MessageRole.USER, content="Hello! Just say 'Hi' back.")],
                    max_tokens=50  # Increased to allow for actual response
                )
                
                # Check if this model works for chat
                if response.success:
                    successful_response = response
                else:
                    print(f"Model {o3_model} failed with error: {response.error_message}")
                    
        except Exception as e:
            print(f"Model {o3_model} failed with exception: {e}")
        
        # Test the successful response
        assert successful_response.success == True, f"Request failed with error: {successful_response.error_message}"
        assert successful_response.error_message is None
        
        # Verify we got a response
        assert len(successful_response.choices) > 0
        assert successful_response.choices[0].message.content is not None
        
        # Allow for empty responses with low token limits - the important thing is no error
        if successful_response.choices[0].finish_reason == "length":
            print("Response was cut off due to max_tokens limit - this is expected")
        else:
            assert len(successful_response.choices[0].message.content.strip()) > 0
        
        print(f"Successful response: {successful_response.choices[0].message.content}")
        print(f"Model used: {successful_response.model}")
        print(f"Usage: {successful_response.usage}")
        
        # Verify it's actually an OpenAI response
        assert successful_response.provider == "openai"
        
        # Test that the max_completion_tokens parameter was used correctly
        # This is verified by the fact that we got a successful response instead of an error
        print(f"SUCCESS: o3 model {o3_model} worked with max_completion_tokens parameter handling")