"""
Dynamic model discovery from provider APIs.

This module provides functionality to query provider APIs for available models
instead of relying on hardcoded lists.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Cache for model lists to avoid hitting APIs too frequently
_model_cache: dict[str, dict[str, Any]] = {}
_cache_duration = timedelta(hours=1)  # Cache models for 1 hour


@dataclass
class ModelDiscoveryResult:
    """Result of model discovery process."""

    # Discovery result
    found_provider: str | None = None
    model_found: bool = False

    # All models found for each provider
    openai_models: set[str] | None = None
    anthropic_models: set[str] | None = None
    openrouter_models: set[str] | None = None

    # Errors encountered during discovery
    errors: dict[str, str] | None = None

    def get_all_models(self) -> dict[str, set[str]]:
        """Get all models organized by provider."""
        result = {}
        if self.openai_models is not None:
            result["openai"] = self.openai_models
        if self.anthropic_models is not None:
            result["anthropic"] = self.anthropic_models
        if self.openrouter_models is not None:
            result["openrouter"] = self.openrouter_models
        return result

    def get_total_models_found(self) -> int:
        """Get total number of models found across all providers."""
        total = 0
        if self.openai_models:
            total += len(self.openai_models)
        if self.anthropic_models:
            total += len(self.anthropic_models)
        if self.openrouter_models:
            total += len(self.openrouter_models)
        return total


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
) -> ModelDiscoveryResult:
    """
    Detect provider from model name using live API queries.

    Args:
        model: The model name to check
        api_keys: Dictionary of API keys {"openai": "sk-...", "anthropic": "sk-ant-..."}

    Returns:
        ModelDiscoveryResult with discovery information
    """
    if not api_keys:
        api_keys = {}

    result = ModelDiscoveryResult(errors={})

    # Handle provider-prefixed models (e.g., "openai/o3", "anthropic/claude-3-sonnet")
    preferred_provider = None
    base_model = model

    if "/" in model:
        parts = model.split("/", 1)
        if len(parts) == 2:
            provider_prefix, base_model = parts
            if provider_prefix == "openai":
                preferred_provider = "openai"
            elif provider_prefix == "anthropic":
                preferred_provider = "anthropic"

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

        # Process results and store all model information
        for provider_name, models_result in zip(provider_names, results, strict=False):
            if isinstance(models_result, Exception):
                logger.debug(f"Failed to check {provider_name} for model {model}: {models_result}")
                if result.errors is not None:
                    result.errors[provider_name] = str(models_result)
                continue

            # Store models in result
            if provider_name == "openai" and isinstance(models_result, set):
                result.openai_models = models_result
            elif provider_name == "anthropic" and isinstance(models_result, set):
                result.anthropic_models = models_result
            elif provider_name == "openrouter" and isinstance(models_result, set):
                result.openrouter_models = models_result

        # Determine the best provider to use
        if preferred_provider and not result.model_found:
            # Check if base model exists in preferred provider
            provider_models = None
            if preferred_provider == "openai" and result.openai_models:
                provider_models = result.openai_models
            elif preferred_provider == "anthropic" and result.anthropic_models:
                provider_models = result.anthropic_models

            if provider_models and base_model in provider_models:
                result.found_provider = preferred_provider
                result.model_found = True
            elif result.openrouter_models and model in result.openrouter_models:
                # Fallback to OpenRouter if base model not found in preferred provider
                result.found_provider = "openrouter"
                result.model_found = True

        # For models without provider prefix, use original logic
        if not result.model_found:
            for provider_name, models_result in zip(provider_names, results, strict=False):
                if isinstance(models_result, Exception):
                    continue

                # Check if our target model was found
                if isinstance(models_result, set) and model in models_result:
                    result.found_provider = provider_name
                    result.model_found = True
                    break

    except Exception as e:
        logger.debug(f"Failed to run dynamic discovery for model {model}: {e}")
        if result.errors is not None:
            result.errors["general"] = str(e)

    return result


def detect_provider_from_model_sync(
    model: str,
    api_keys: dict[str, str] | None = None
) -> ModelDiscoveryResult:
    """Synchronous version of detect_provider_from_model_async."""
    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        # We're in an async context, but need to run in sync mode
        # Create a new event loop in a thread
        import concurrent.futures

        def run_in_thread() -> ModelDiscoveryResult:
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



