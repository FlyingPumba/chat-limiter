"""
Dynamic model discovery from provider APIs.

This module provides functionality to query provider APIs for available models
instead of relying on hardcoded lists.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Cache for model lists to avoid hitting APIs too frequently
_model_cache: dict[str, dict[str, Any]] = {}
_cache_duration = timedelta(hours=1)  # Cache models for 1 hour


class ModelDiscovery:
    """Dynamic model discovery from provider APIs."""

    @staticmethod
    async def get_openai_models(api_key: str) -> set[str]:
        """Get available OpenAI models from the API."""
        cache_key = f"openai_models_{hash(api_key)}"

        # Check cache first
        if _model_cache.get(cache_key):
            cache_entry = _model_cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < _cache_duration:
                return cache_entry["models"]  # type: ignore[no-any-return]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                models = set()

                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Filter for chat completion models
                    if any(keyword in model_id.lower() for keyword in ["gpt", "chat"]):
                        models.add(model_id)

                # Cache the result
                _model_cache[cache_key] = {
                    "models": models,
                    "timestamp": datetime.now()
                }

                logger.info(f"Retrieved {len(models)} OpenAI models from API")
                return models

        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            raise

    @staticmethod
    async def get_anthropic_models(api_key: str) -> set[str]:
        """Get available Anthropic models from the API."""
        cache_key = f"anthropic_models_{hash(api_key)}"

        # Check cache first
        if _model_cache.get(cache_key):
            cache_entry = _model_cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < _cache_duration:
                return cache_entry["models"]  # type: ignore[no-any-return]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                models = set()

                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Filter for Claude models
                    if "claude" in model_id.lower():
                        models.add(model_id)

                # Cache the result
                _model_cache[cache_key] = {
                    "models": models,
                    "timestamp": datetime.now()
                }

                logger.info(f"Retrieved {len(models)} Anthropic models from API")
                return models

        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")
            raise

    @staticmethod
    async def get_openrouter_models(api_key: str | None = None) -> set[str]:
        """Get available OpenRouter models from the API."""
        cache_key = "openrouter_models"

        # Check cache first
        if _model_cache.get(cache_key):
            cache_entry = _model_cache[cache_key]
            if datetime.now() - cache_entry["timestamp"] < _cache_duration:
                return cache_entry["models"]  # type: ignore[no-any-return]

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers=headers,
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                models = set()

                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if model_id:
                        models.add(model_id)

                # Cache the result
                _model_cache[cache_key] = {
                    "models": models,
                    "timestamp": datetime.now()
                }

                logger.info(f"Retrieved {len(models)} OpenRouter models from API")
                return models

        except Exception as e:
            logger.warning(f"Failed to fetch OpenRouter models: {e}")
            raise

    @staticmethod
    def get_openai_models_sync(api_key: str) -> set[str]:
        """Synchronous version of get_openai_models."""
        return asyncio.run(ModelDiscovery.get_openai_models(api_key))

    @staticmethod
    def get_anthropic_models_sync(api_key: str) -> set[str]:
        """Synchronous version of get_anthropic_models."""
        return asyncio.run(ModelDiscovery.get_anthropic_models(api_key))

    @staticmethod
    def get_openrouter_models_sync(api_key: str | None = None) -> set[str]:
        """Synchronous version of get_openrouter_models."""
        return asyncio.run(ModelDiscovery.get_openrouter_models(api_key))


async def detect_provider_from_model_async(
    model: str,
    api_keys: dict[str, str] | None = None
) -> str | None:
    """
    Detect provider from model name using live API queries.

    Args:
        model: The model name to check
        api_keys: Dictionary of API keys {"openai": "sk-...", "anthropic": "sk-ant-..."}

    Returns:
        Provider name or None if not found
    """
    if not api_keys:
        api_keys = {}

    # First try simple pattern matching for known formats
    if "/" in model:  # OpenRouter format
        return "openrouter"

    # Create all tasks
    tasks = []

    if api_keys.get("openai"):
        tasks.append(("openai", ModelDiscovery.get_openai_models(api_keys["openai"])))

    if api_keys.get("anthropic"):
        tasks.append(("anthropic", ModelDiscovery.get_anthropic_models(api_keys["anthropic"])))

    if api_keys.get("openrouter"):
        tasks.append(("openrouter", ModelDiscovery.get_openrouter_models(api_keys["openrouter"])))
    else:
        # OpenRouter doesn't require API key for model listing
        tasks.append(("openrouter", ModelDiscovery.get_openrouter_models()))

    # Use asyncio.gather to run all tasks concurrently and properly handle them
    try:
        # Extract just the coroutines for gather
        coroutines = [task[1] for task in tasks]
        provider_names = [task[0] for task in tasks]
        
        # Wait for all results
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Check results in order
        for provider_name, result in zip(provider_names, results):
            if isinstance(result, Exception):
                logger.debug(f"Failed to check {provider_name} for model {model}: {result}")
                continue
            if model in result:
                return provider_name
                
    except Exception as e:
        logger.debug(f"Failed to run dynamic discovery for model {model}: {e}")

    return None


def detect_provider_from_model_sync(
    model: str,
    api_keys: dict[str, str] | None = None
) -> str | None:
    """Synchronous version of detect_provider_from_model_async."""
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, but need to run in sync mode
        # Create a new event loop in a thread
        import concurrent.futures
        
        def run_in_thread():
            return asyncio.run(detect_provider_from_model_async(model, api_keys))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=30)  # 30 second timeout
            
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(detect_provider_from_model_async(model, api_keys))


def clear_model_cache() -> None:
    """Clear the model cache to force fresh API queries."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")



