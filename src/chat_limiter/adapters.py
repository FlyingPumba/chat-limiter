"""
Provider-specific adapters for converting between our unified types and provider APIs.
"""

import time
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

        # Build request
        openai_request: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        # Add optional parameters
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
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

        return openai_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse OpenAI response."""
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

        # Build request
        anthropic_request: dict[str, Any] = {
            "model": request.model,
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

        return anthropic_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse Anthropic response."""
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

        # Build request
        openrouter_request: dict[str, Any] = {
            "model": request.model,
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

        return openrouter_request

    def parse_response(
        self,
        response_data: dict[str, Any],
        original_request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Parse OpenRouter response (similar to OpenAI)."""
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
