"""
Provider-specific adapters for converting between our unified types and provider APIs.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any

from .providers import Provider
from .types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
)


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific adapters."""

    def is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model (o1, o3, o4 series)."""
        # Handle prefixed models (e.g., "openai/o3-mini")
        if "/" in model_name:
            # Extract the base model name after the "/"
            base_model = model_name.split("/", 1)[1]
            return base_model.startswith(("o1", "o3", "o4", "gpt-5"))

        # Handle non-prefixed models
        return model_name.startswith(("o1", "o3", "o4", "gpt-5"))

    @abstractmethod
    def format_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Convert our request format to provider-specific format."""
        pass

    @abstractmethod
    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Convert provider response to our unified format."""
        pass

    @abstractmethod
    def get_endpoint(self) -> str:
        """Get the API endpoint for this provider."""
        pass


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API."""

    def format_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Convert to OpenAI format."""
        # Convert messages
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        model = request.model.strip()
        if model.startswith("openai/"):
            # Remove the "openai/" prefix, since we are already using the OpenAI API
            model = model.split("openai/", 1)[1]

        # Build request
        openai_request: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Add optional parameters
        if request.max_tokens is not None:
            # Use max_completion_tokens for reasoning models (o1, o3, o4)
            if self.is_reasoning_model(model):
                openai_request["max_completion_tokens"] = request.max_tokens
            else:
                openai_request["max_tokens"] = request.max_tokens

        # Handle temperature for reasoning models
        if self.is_reasoning_model(model):
            # For reasoning models, default to temperature=1
            default_temperature = 1.0

            if request.temperature is not None:
                # If user provided a different temperature, warn them and use temperature=1
                if request.temperature != default_temperature:
                    warnings.warn(
                        f"WARNING: Model '{model}' is a reasoning model that requires temperature=1. "
                        f"Your specified temperature={request.temperature} will be overridden to temperature=1.",
                        UserWarning
                    )
                    print(f"WARNING: Model '{model}' is a reasoning model that requires temperature=1. "
                          f"Your specified temperature={request.temperature} will be overridden to temperature=1.")

            # Always use temperature=1 for reasoning models
            openai_request["temperature"] = default_temperature
        else:
            # For non-reasoning models, use the provided temperature
            if request.temperature is not None:
                openai_request["temperature"] = request.temperature

        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.stop is not None:
            openai_request["stop"] = request.stop
        if request.stream:
            openai_request["stream"] = request.stream
        if request.frequency_penalty is not None:
            openai_request["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            openai_request["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            openai_request["seed"] = request.seed

        # Add reasoning parameter for thinking models
        if (request.reasoning_effort is not None and
            self.is_reasoning_model(model)):
            openai_request["reasoning"] = {"effort": request.reasoning_effort}

        return openai_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse OpenAI response."""
        # Check for errors first
        success = True
        error_message = None

        if "error" in response_data:
            success = False
            error_data = response_data["error"]
            error_message = error_data.get("message", "Unknown error")

        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})

            # Handle both string and content-block formats
            raw_content = message_data.get("content", "")
            content_text = ""
            if isinstance(raw_content, str) and raw_content:
                content_text = raw_content
            elif isinstance(raw_content, list) and raw_content:
                # Newer OpenAI responses may return a list of content blocks
                parts: list[str] = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    # Prefer explicit output fields used by reasoning models
                    output_text_val = block.get("output_text")
                    if isinstance(output_text_val, str) and output_text_val:
                        parts.append(output_text_val)
                        continue
                    # Fallbacks
                    text_val = block.get("text")
                    if isinstance(text_val, str) and text_val:
                        parts.append(text_val)
                        continue
                    content_val = block.get("content")
                    if isinstance(content_val, str) and content_val:
                        parts.append(content_val)
                content_text = "".join(parts)
            # Choice-level fallback sometimes present in reasoning responses
            if not content_text:
                choice_level_output = choice_data.get("output_text")
                if isinstance(choice_level_output, str) and choice_level_output:
                    content_text = choice_level_output

            message = Message(
                role=MessageRole(message_data.get("role", "assistant")),
                content=content_text
            )
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)

        # Parse usage
        usage = None
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            model=response_data.get("model", original_request.model),
            choices=choices,
            usage=usage,
            created=response_data.get("created"),
            success=success,
            error_message=error_message,
            provider="openai",
            raw_response=response_data
        )

    def get_endpoint(self) -> str:
        return "/chat/completions"


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API."""

    def format_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Convert to Anthropic format."""
        # Anthropic has a different message format
        messages: list[dict[str, Any]] = []
        system_message: str | None = None

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic handles system messages separately
                system_message = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        model = request.model.strip()
        if model.startswith("anthropic/"):
            # Remove the "anthropic/" prefix, since we are already using the Anthropic API
            model = model.split("anthropic/", 1)[1]

        # Build request
        anthropic_request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,  # Required for Anthropic
        }

        # Add system message if present
        if system_message:
            anthropic_request["system"] = system_message

        # Add optional parameters
        if request.temperature is not None:
            anthropic_request["temperature"] = request.temperature
        if request.top_p is not None:
            anthropic_request["top_p"] = request.top_p
        if request.stop is not None:
            anthropic_request["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )
        if request.stream:
            anthropic_request["stream"] = request.stream
        if request.top_k is not None:
            anthropic_request["top_k"] = request.top_k
        if request.seed is not None:
            anthropic_request["seed"] = request.seed

        return anthropic_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse Anthropic response."""
        # Check for errors first
        success = True
        error_message = None

        if "error" in response_data:
            success = False
            error_data = response_data["error"]
            error_message = error_data.get("message", "Unknown error")

        # Anthropic returns content differently
        content_blocks = response_data.get("content", [])
        content = ""
        if content_blocks:
            # Extract text from content blocks
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

        message = Message(
            role=MessageRole.ASSISTANT,
            content=content
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason=response_data.get("stop_reason")
        )

        # Parse usage
        usage = None
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
            )

        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            model=response_data.get("model", original_request.model),
            choices=[choice],
            usage=usage,
            created=int(time.time()),  # Anthropic doesn't provide created timestamp
            success=success,
            error_message=error_message,
            provider="anthropic",
            raw_response=response_data
        )

    def get_endpoint(self) -> str:
        return "/messages"


class OpenRouterAdapter(ProviderAdapter):
    """Adapter for OpenRouter API."""

    def format_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Convert to OpenRouter format (similar to OpenAI)."""
        # OpenRouter uses OpenAI-compatible format
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        model = request.model.strip()

        # Build request
        openrouter_request: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Add optional parameters
        if request.max_tokens is not None:
            openrouter_request["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            openrouter_request["temperature"] = request.temperature
        if request.top_p is not None:
            openrouter_request["top_p"] = request.top_p
        if request.stop is not None:
            openrouter_request["stop"] = request.stop
        if request.stream:
            openrouter_request["stream"] = request.stream
        if request.frequency_penalty is not None:
            openrouter_request["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            openrouter_request["presence_penalty"] = request.presence_penalty
        if request.top_k is not None:
            openrouter_request["top_k"] = request.top_k
        if request.seed is not None:
            openrouter_request["seed"] = request.seed

        # Add reasoning parameter for thinking models
        if (request.reasoning_effort is not None and
            self.is_reasoning_model(model)):
            openrouter_request["reasoning"] = {"effort": request.reasoning_effort}

        # Add provider routing if specified
        if request.providers is not None:
            openrouter_request["provider"] = {
                "order": request.providers,
                "allow_fallbacks": False
            }

        return openrouter_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse OpenRouter response (similar to OpenAI)."""
        # Check for errors first
        success = True
        error_message = None

        if "error" in response_data:
            success = False
            error_data = response_data["error"]
            error_message = error_data.get("message", "Unknown error")

        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            message = Message(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", "")
            )
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)

        # Parse usage
        usage = None
        if "usage" in response_data:
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )

        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            model=response_data.get("model", original_request.model),
            choices=choices,
            usage=usage,
            created=response_data.get("created"),
            success=success,
            error_message=error_message,
            provider="openrouter",
            raw_response=response_data
        )

    def get_endpoint(self) -> str:
        return "/chat/completions"


# Provider adapter registry
PROVIDER_ADAPTERS = {
    Provider.OPENAI: OpenAIAdapter(),
    Provider.ANTHROPIC: AnthropicAdapter(),
    Provider.OPENROUTER: OpenRouterAdapter(),
}


def get_adapter(provider: Provider) -> ProviderAdapter:
    """Get the appropriate adapter for a provider."""
    return PROVIDER_ADAPTERS[provider]
