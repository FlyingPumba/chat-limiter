"""
Tests for the high-level types module.
"""


from chat_limiter.types import (
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
    detect_provider_from_model,
)


class TestMessageRole:
    def test_message_role_values(self):
        """Test that MessageRole has the expected values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"


class TestMessage:
    def test_message_creation(self):
        """Test Message creation with valid data."""
        message = Message(role=MessageRole.USER, content="Hello!")
        assert message.role == MessageRole.USER
        assert message.content == "Hello!"

    def test_message_creation_with_string_role(self):
        """Test Message creation with string role."""
        message = Message(role="user", content="Hello!")  # type: ignore
        assert message.role == "user"
        assert message.content == "Hello!"


class TestChatCompletionRequest:
    def test_request_creation_minimal(self):
        """Test ChatCompletionRequest creation with minimal data."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        request = ChatCompletionRequest(model="gpt-4o", messages=messages)

        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.messages[0].content == "Hello!"
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.stream is False

    def test_request_creation_full(self):
        """Test ChatCompletionRequest creation with all parameters."""
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stop=["\\n"],
            stream=True,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_k=40,
        )

        assert request.model == "gpt-4o"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stop == ["\\n"]
        assert request.stream is True
        assert request.frequency_penalty == 0.5
        assert request.presence_penalty == 0.3
        assert request.top_k == 40

    def test_request_validation(self):
        """Test ChatCompletionRequest validation."""
        # Should work with valid data
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        request = ChatCompletionRequest(model="gpt-4o", messages=messages)
        assert request.model == "gpt-4o"

    def test_request_with_multiple_messages(self):
        """Test request with multiple messages."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            Message(role=MessageRole.USER, content="How are you?"),
        ]
        request = ChatCompletionRequest(model="gpt-4o", messages=messages)
        assert len(request.messages) == 4
        assert request.messages[0].role == MessageRole.SYSTEM
        assert request.messages[-1].content == "How are you?"


class TestUsage:
    def test_usage_creation(self):
        """Test Usage creation."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class TestChoice:
    def test_choice_creation(self):
        """Test Choice creation."""
        message = Message(role=MessageRole.ASSISTANT, content="Hello!")
        choice = Choice(index=0, message=message, finish_reason="stop")
        assert choice.index == 0
        assert choice.message.content == "Hello!"
        assert choice.finish_reason == "stop"

    def test_choice_creation_minimal(self):
        """Test Choice creation with minimal data."""
        message = Message(role=MessageRole.ASSISTANT, content="Hello!")
        choice = Choice(index=0, message=message)
        assert choice.index == 0
        assert choice.message.content == "Hello!"
        assert choice.finish_reason is None


class TestChatCompletionResponse:
    def test_response_creation_minimal(self):
        """Test ChatCompletionResponse creation with minimal data."""
        message = Message(role=MessageRole.ASSISTANT, content="Hello!")
        choice = Choice(index=0, message=message)
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="gpt-4o",
            choices=[choice]
        )

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.usage is None
        assert response.created is None
        assert response.provider is None

    def test_response_creation_full(self):
        """Test ChatCompletionResponse creation with all data."""
        message = Message(role=MessageRole.ASSISTANT, content="Hello!")
        choice = Choice(index=0, message=message, finish_reason="stop")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="gpt-4o",
            choices=[choice],
            usage=usage,
            created=1234567890,
            provider="openai",
            raw_response={"test": "data"}
        )

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o"
        assert response.usage.total_tokens == 15
        assert response.created == 1234567890
        assert response.provider == "openai"
        assert response.raw_response == {"test": "data"}

    def test_response_with_multiple_choices(self):
        """Test response with multiple choices."""
        choices = [
            Choice(index=0, message=Message(role=MessageRole.ASSISTANT, content="First")),
            Choice(index=1, message=Message(role=MessageRole.ASSISTANT, content="Second")),
        ]
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            model="gpt-4o",
            choices=choices
        )

        assert len(response.choices) == 2
        assert response.choices[0].message.content == "First"
        assert response.choices[1].message.content == "Second"


class TestProviderDetection:
    def test_detect_openai_models(self):
        """Test detection of OpenAI models."""
        assert detect_provider_from_model("gpt-4o") == "openai"
        assert detect_provider_from_model("gpt-4o-mini") == "openai"
        assert detect_provider_from_model("gpt-3.5-turbo") == "openai"

    def test_detect_anthropic_models(self):
        """Test detection of Anthropic models."""
        assert detect_provider_from_model("claude-3-5-sonnet-20241022") == "anthropic"
        assert detect_provider_from_model("claude-3-opus-20240229") == "anthropic"
        assert detect_provider_from_model("claude-3-haiku-20240307") == "anthropic"

    def test_detect_openrouter_models(self):
        """Test detection of OpenRouter models."""
        assert detect_provider_from_model("openai/gpt-4o") == "openrouter"
        assert detect_provider_from_model("anthropic/claude-3-opus") == "openrouter"
        assert detect_provider_from_model("meta-llama/llama-3.1-405b-instruct") == "openrouter"

    def test_detect_openrouter_format(self):
        """Test detection of OpenRouter format (provider/model)."""
        assert detect_provider_from_model("some-provider/some-model") == "openrouter"
        assert detect_provider_from_model("custom/model-name") == "openrouter"

    def test_detect_unknown_model(self):
        """Test detection of unknown models."""
        assert detect_provider_from_model("unknown-model") is None
        assert detect_provider_from_model("") is None
        assert detect_provider_from_model("random-text") is None

    def test_model_sets_coverage(self):
        """Test that model sets contain expected models."""
        # Test some specific models we know should be there
        assert "gpt-4o" in OPENAI_MODELS
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_MODELS
        assert "openai/gpt-4o" in OPENROUTER_MODELS

        # Test that sets are not empty
        assert len(OPENAI_MODELS) > 0
        assert len(ANTHROPIC_MODELS) > 0
        assert len(OPENROUTER_MODELS) > 0

        # Test that sets don't overlap (except for OpenRouter having provider prefixes)
        assert len(OPENAI_MODELS & ANTHROPIC_MODELS) == 0
