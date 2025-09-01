"""
Tests for the provider adapters module.
"""


from chat_limiter.adapters import (
    AnthropicAdapter,
    OpenAIAdapter,
    OpenRouterAdapter,
    get_adapter,
)
from chat_limiter.providers import Provider
from chat_limiter.types import (
    ChatCompletionRequest,
    Message,
    MessageRole,
)


class TestOpenAIAdapter:
    def test_format_request_basic(self):
        """Test OpenAI request formatting with basic parameters."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        formatted = adapter.format_request(request)

        assert formatted["model"] == "gpt-4o"
        assert len(formatted["messages"]) == 1
        assert formatted["messages"][0]["role"] == "user"
        assert formatted["messages"][0]["content"] == "Hello!"

    def test_format_request_full(self):
        """Test OpenAI request formatting with all parameters."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop=["\\n"],
            stream=True,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        formatted = adapter.format_request(request)

        assert formatted["model"] == "gpt-4o"
        assert formatted["max_tokens"] == 100
        assert formatted["temperature"] == 0.7
        assert formatted["top_p"] == 0.9
        assert formatted["stop"] == ["\\n"]
        assert formatted["stream"] is True
        assert formatted["frequency_penalty"] == 0.5
        assert formatted["presence_penalty"] == 0.3

    def test_format_request_multiple_messages(self):
        """Test OpenAI request formatting with multiple messages."""
        adapter = OpenAIAdapter()
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
        ]
        request = ChatCompletionRequest(model="gpt-4o", messages=messages)

        formatted = adapter.format_request(request)

        assert len(formatted["messages"]) == 3
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][2]["role"] == "assistant"

    def test_parse_response_basic(self):
        """Test OpenAI response parsing."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "created": 1234567890
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.id == "chatcmpl-123"
        assert parsed.model == "gpt-4o-2024-08-06"
        assert parsed.provider == "openai"
        assert len(parsed.choices) == 1
        assert parsed.choices[0].message.content == "Hello there!"
        assert parsed.choices[0].finish_reason == "stop"
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.total_tokens == 15
        assert parsed.created == 1234567890
        assert parsed.success == True
        assert parsed.error_message is None

    def test_parse_response_no_usage(self):
        """Test OpenAI response parsing without usage information."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    }
                }
            ]
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.usage is None
        assert parsed.model == "gpt-4o"  # Falls back to request model

    def test_parse_response_with_error(self):
        """Test OpenAI response parsing with error."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "error": {
                "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "code": "unsupported_parameter"
            }
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.success == False
        assert parsed.error_message == "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
        assert parsed.model == "gpt-4o"  # Falls back to request model
        assert len(parsed.choices) == 0  # No choices in error response

    def test_is_reasoning_model(self):
        """Test reasoning model detection."""
        adapter = OpenAIAdapter()
        
        # Test reasoning models
        assert adapter.is_reasoning_model("o1-preview") == True
        assert adapter.is_reasoning_model("o1-mini") == True
        assert adapter.is_reasoning_model("o3-mini") == True
        assert adapter.is_reasoning_model("o3-2024-01-01") == True
        assert adapter.is_reasoning_model("o4-preview") == True
        
        # Test non-reasoning models
        assert adapter.is_reasoning_model("gpt-4o") == False
        assert adapter.is_reasoning_model("gpt-4o-mini") == False
        assert adapter.is_reasoning_model("gpt-4-turbo") == False
        assert adapter.is_reasoning_model("gpt-3.5-turbo") == False
        
        # Test prefixed reasoning models (OpenRouter format)
        assert adapter.is_reasoning_model("openai/o1-preview") == True
        assert adapter.is_reasoning_model("openai/o3-mini") == True
        assert adapter.is_reasoning_model("openai/o4-preview") == True
        
        # Test prefixed non-reasoning models
        assert adapter.is_reasoning_model("openai/gpt-4o") == False
        assert adapter.is_reasoning_model("anthropic/claude-3-sonnet") == False

    def test_format_request_reasoning_model(self):
        """Test request formatting for reasoning models uses max_completion_tokens."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            max_tokens=100
        )

        formatted = adapter.format_request(request)

        # Should use max_completion_tokens for reasoning models
        assert "max_completion_tokens" in formatted
        assert formatted["max_completion_tokens"] == 100
        assert "max_tokens" not in formatted

    def test_format_request_non_reasoning_model(self):
        """Test request formatting for non-reasoning models uses max_tokens."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            max_tokens=100
        )

        formatted = adapter.format_request(request)

        # Should use max_tokens for non-reasoning models
        assert "max_tokens" in formatted
        assert formatted["max_tokens"] == 100
        assert "max_completion_tokens" not in formatted

    def test_get_endpoint(self):
        """Test OpenAI endpoint."""
        adapter = OpenAIAdapter()
        assert adapter.get_endpoint() == "/chat/completions"

    def test_reasoning_effort_for_reasoning_model(self):
        """Test that reasoning_effort is added for reasoning models."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort="medium"
        )

        formatted = adapter.format_request(request)

        assert formatted["reasoning"] == {"effort": "medium"}

    def test_reasoning_effort_for_non_reasoning_model(self):
        """Test that reasoning_effort is NOT added for non-reasoning models."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort="medium"
        )

        formatted = adapter.format_request(request)

        assert "reasoning" not in formatted

    def test_reasoning_effort_none_value(self):
        """Test that reasoning_effort=None doesn't add reasoning parameter."""
        adapter = OpenAIAdapter()
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort=None
        )

        formatted = adapter.format_request(request)

        assert "reasoning" not in formatted

    def test_reasoning_effort_all_valid_values(self):
        """Test all valid reasoning_effort values."""
        adapter = OpenAIAdapter()
        
        for effort in ["low", "medium", "high"]:
            request = ChatCompletionRequest(
                model="o3-mini",
                messages=[Message(role=MessageRole.USER, content="Hello!")],
                reasoning_effort=effort
            )

            formatted = adapter.format_request(request)

            assert formatted["reasoning"] == {"effort": effort}


class TestAnthropicAdapter:
    def test_format_request_basic(self):
        """Test Anthropic request formatting with basic parameters."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            max_tokens=100
        )

        formatted = adapter.format_request(request)

        assert formatted["model"] == "claude-3-sonnet-20240229"
        assert formatted["max_tokens"] == 100
        assert len(formatted["messages"]) == 1
        assert formatted["messages"][0]["role"] == "user"
        assert formatted["messages"][0]["content"] == "Hello!"

    def test_format_request_with_system_message(self):
        """Test Anthropic request formatting with system message."""
        adapter = AnthropicAdapter()
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
        ]
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=messages,
            max_tokens=100
        )

        formatted = adapter.format_request(request)

        assert formatted["system"] == "You are helpful."
        assert len(formatted["messages"]) == 1  # System message excluded
        assert formatted["messages"][0]["role"] == "user"

    def test_format_request_default_max_tokens(self):
        """Test Anthropic request formatting with default max_tokens."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        formatted = adapter.format_request(request)

        assert formatted["max_tokens"] == 1024  # Default value

    def test_format_request_anthropic_specific(self):
        """Test Anthropic-specific parameters."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop="\\n",
            top_k=40,
        )

        formatted = adapter.format_request(request)

        assert formatted["temperature"] == 0.7
        assert formatted["top_p"] == 0.9
        assert formatted["stop_sequences"] == ["\\n"]  # Converted to list
        assert formatted["top_k"] == 40

    def test_format_request_stop_list(self):
        """Test Anthropic stop sequences with list input."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            stop=["\\n", "END"]
        )

        formatted = adapter.format_request(request)

        assert formatted["stop_sequences"] == ["\\n", "END"]

    def test_parse_response_basic(self):
        """Test Anthropic response parsing."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "id": "msg_123",
            "model": "claude-3-sonnet-20240229",
            "content": [
                {
                    "type": "text",
                    "text": "Hello there!"
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.id == "msg_123"
        assert parsed.model == "claude-3-sonnet-20240229"
        assert parsed.provider == "anthropic"
        assert len(parsed.choices) == 1
        assert parsed.choices[0].message.content == "Hello there!"
        assert parsed.choices[0].finish_reason == "end_turn"
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 5
        assert parsed.usage.total_tokens == 15
        assert parsed.success == True
        assert parsed.error_message is None

    def test_parse_response_multiple_content_blocks(self):
        """Test Anthropic response parsing with multiple content blocks."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "id": "msg_123",
            "content": [
                {
                    "type": "text",
                    "text": "Hello "
                },
                {
                    "type": "text",
                    "text": "there!"
                }
            ]
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.choices[0].message.content == "Hello there!"

    def test_parse_response_with_error(self):
        """Test Anthropic response parsing with error."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "error": {
                "message": "Invalid parameter: temperature must be between 0 and 1",
                "type": "invalid_request_error"
            }
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.success == False
        assert parsed.error_message == "Invalid parameter: temperature must be between 0 and 1"
        assert parsed.model == "claude-3-sonnet-20240229"  # Falls back to request model
        assert len(parsed.choices) == 1  # Anthropic still creates a choice with empty content
        assert parsed.choices[0].message.content == ""

    def test_get_endpoint(self):
        """Test Anthropic endpoint."""
        adapter = AnthropicAdapter()
        assert adapter.get_endpoint() == "/messages"

    def test_reasoning_effort_not_supported(self):
        """Test that Anthropic adapter doesn't add reasoning parameter (not supported)."""
        adapter = AnthropicAdapter()
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort="medium"
        )

        formatted = adapter.format_request(request)

        # Anthropic doesn't support reasoning effort
        assert "reasoning" not in formatted


class TestOpenRouterAdapter:
    def test_format_request_basic(self):
        """Test OpenRouter request formatting (similar to OpenAI)."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        formatted = adapter.format_request(request)

        assert formatted["model"] == "openai/gpt-4o"
        assert len(formatted["messages"]) == 1
        assert formatted["messages"][0]["role"] == "user"
        assert formatted["messages"][0]["content"] == "Hello!"

    def test_format_request_with_top_k(self):
        """Test OpenRouter request formatting with top_k parameter."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            top_k=40,
        )

        formatted = adapter.format_request(request)

        assert formatted["top_k"] == 40

    def test_parse_response_basic(self):
        """Test OpenRouter response parsing (similar to OpenAI)."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.provider == "openrouter"
        assert parsed.id == "chatcmpl-123"
        assert parsed.choices[0].message.content == "Hello there!"
        assert parsed.success == True
        assert parsed.error_message is None

    def test_parse_response_with_error(self):
        """Test OpenRouter response parsing with error."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        response_data = {
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }

        parsed = adapter.parse_response(response_data, request)

        assert parsed.success == False
        assert parsed.error_message == "Model not found"
        assert parsed.model == "openai/gpt-4o"  # Falls back to request model
        assert len(parsed.choices) == 0  # No choices in error response

    def test_get_endpoint(self):
        """Test OpenRouter endpoint."""
        adapter = OpenRouterAdapter()
        assert adapter.get_endpoint() == "/chat/completions"

    def test_reasoning_effort_for_reasoning_model(self):
        """Test that reasoning_effort is added for reasoning models in OpenRouter."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/o3-mini",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort="high"
        )

        formatted = adapter.format_request(request)

        assert formatted["reasoning"] == {"effort": "high"}

    def test_reasoning_effort_for_non_reasoning_model(self):
        """Test that reasoning_effort is NOT added for non-reasoning models in OpenRouter."""
        adapter = OpenRouterAdapter()
        request = ChatCompletionRequest(
            model="openai/gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            reasoning_effort="medium"
        )

        formatted = adapter.format_request(request)

        assert "reasoning" not in formatted


class TestGetAdapter:
    def test_get_openai_adapter(self):
        """Test getting OpenAI adapter."""
        adapter = get_adapter(Provider.OPENAI)
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_anthropic_adapter(self):
        """Test getting Anthropic adapter."""
        adapter = get_adapter(Provider.ANTHROPIC)
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_openrouter_adapter(self):
        """Test getting OpenRouter adapter."""
        adapter = get_adapter(Provider.OPENROUTER)
        assert isinstance(adapter, OpenRouterAdapter)

    def test_adapter_registry_completeness(self):
        """Test that all providers have adapters."""
        for provider in Provider:
            adapter = get_adapter(provider)
            assert adapter is not None
