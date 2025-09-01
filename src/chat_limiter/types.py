"""
Type definitions for chat completion requests and responses.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Message roles supported across providers."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A chat message that works across all providers."""

    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    """High-level chat completion request."""

    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False

    # Provider-specific parameters (will be filtered per provider)
    frequency_penalty: float | None = None  # OpenAI
    presence_penalty: float | None = None   # OpenAI
    top_k: int | None = None               # Anthropic
    reasoning_effort: str | None = None    # OpenAI/OpenRouter reasoning models


@dataclass
class Usage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice:
    """A completion choice."""

    index: int
    message: Message
    finish_reason: str | None = None


@dataclass
class ChatCompletionResponse:
    """High-level chat completion response."""

    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    created: int | None = None

    # Error information
    success: bool = True
    error_message: str | None = None

    # Provider-specific metadata
    provider: str | None = None
    raw_response: dict[str, Any] | None = None


# Model mappings for each provider
OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
}

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}

OPENROUTER_MODELS = {
    # OpenAI models via OpenRouter
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",

    # Anthropic models via OpenRouter
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",

    # Other providers via OpenRouter
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemini-pro",
    "cohere/command-r-plus",
}

ALL_MODELS = OPENAI_MODELS | ANTHROPIC_MODELS | OPENROUTER_MODELS


def detect_provider_from_model(model: str, use_dynamic_discovery: bool = False, api_keys: dict[str, str] | None = None) -> str | None:
    """
    Detect provider from model name.

    Args:
        model: The model name to check
        use_dynamic_discovery: Whether to use live API queries for model discovery
        api_keys: Dictionary of API keys for dynamic discovery

    Returns:
        Provider name or None if not found
    """
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

    # If we have a preferred provider, check if the base model exists in hardcoded lists
    if preferred_provider:
        if preferred_provider == "openai" and base_model in OPENAI_MODELS:
            return "openai"
        elif preferred_provider == "anthropic" and base_model in ANTHROPIC_MODELS:
            return "anthropic"
        # If base model not found in preferred provider, fall back to checking if full model is in OpenRouter
        elif model in OPENROUTER_MODELS:
            return "openrouter"

    # Check hardcoded lists for fast lookup (for models without provider prefix)
    if model in OPENAI_MODELS:
        return "openai"
    elif model in ANTHROPIC_MODELS:
        return "anthropic"
    elif model in OPENROUTER_MODELS:
        return "openrouter"

    # If dynamic discovery is enabled and we have API keys, try that
    if use_dynamic_discovery and api_keys:
        from .models import detect_provider_from_model_sync
        result = detect_provider_from_model_sync(model, api_keys)
        return result.found_provider

    return None
