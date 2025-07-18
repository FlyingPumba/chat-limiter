"""Live tests for json_mode functionality with gpt-3.5-turbo."""

import json
import os
import pytest

from chat_limiter import ChatLimiter, Message, MessageRole, ChatCompletionRequest, Provider
from chat_limiter.batch import BatchConfig, ChatCompletionBatchProcessor, create_chat_completion_requests


class TestJsonModeLive:
    """Live tests for json_mode functionality."""

    def test_json_mode_assertion_with_anthropic(self):
        """Test that json_mode raises assertion error with Anthropic provider."""
        limiter = ChatLimiter(provider=Provider.ANTHROPIC, api_key="test-key")
        config = BatchConfig(json_mode=True)
        processor = ChatCompletionBatchProcessor(limiter, config)
        
        request = ChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        
        # Should raise AssertionError
        with pytest.raises(AssertionError, match="json_mode is only supported with OpenAI provider"):
            processor.process_item_sync(processor.create_batch_items([request])[0])

    def test_json_mode_assertion_with_openrouter(self):
        """Test that json_mode raises assertion error with OpenRouter provider."""
        limiter = ChatLimiter(provider=Provider.OPENROUTER, api_key="test-key")
        config = BatchConfig(json_mode=True)
        processor = ChatCompletionBatchProcessor(limiter, config)
        
        request = ChatCompletionRequest(
            model="openai/gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        
        # Should raise AssertionError
        with pytest.raises(AssertionError, match="json_mode is only supported with OpenAI provider"):
            processor.process_item_sync(processor.create_batch_items([request])[0])

    @pytest.mark.asyncio
    async def test_json_mode_with_gpt35_turbo(self):
        """Test json_mode with gpt-3.5-turbo produces valid JSON."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Create requests that should produce JSON
        requests = create_chat_completion_requests(
            model="gpt-3.5-turbo",
            prompts=[
                "Generate a JSON object with the following keys: name, age, city. Use realistic values.",
                "Create a JSON object representing a book with title, author, and year_published fields.",
                "Return a JSON object with two fields: greeting and language. Use English values."
            ],
            max_tokens=100
        )

        # Create batch config with json_mode enabled
        config = BatchConfig(
            json_mode=True,
            max_concurrent_requests=3,
            show_progress=False,  # Disable progress for clean test output
            print_prompts=False,
            print_responses=False
        )

        # Process batch with json_mode
        async with ChatLimiter.for_model("gpt-3.5-turbo", api_key=api_key) as limiter:
            processor = ChatCompletionBatchProcessor(limiter, config)
            results = await processor.process_batch(requests)

        # Verify all requests succeeded
        assert len(results) == 3
        for result in results:
            assert result.success, f"Request failed: {result.error_message}"
            assert result.result is not None
            assert len(result.result.choices) > 0
            
            # Extract the response content
            response_content = result.result.choices[0].message.content
            assert response_content is not None
            
            # Verify the response is valid JSON
            try:
                parsed_json = json.loads(response_content)
                assert isinstance(parsed_json, dict), "Response should be a JSON object"
                print(f"✅ Valid JSON response: {parsed_json}")
            except json.JSONDecodeError as e:
                pytest.fail(f"Response is not valid JSON: {response_content}. Error: {e}")

    def test_json_mode_with_gpt35_turbo_sync(self):
        """Test json_mode with gpt-3.5-turbo produces valid JSON (sync version)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Create a simple request
        requests = create_chat_completion_requests(
            model="gpt-3.5-turbo",
            prompts=["Generate a JSON object with keys: status, message, timestamp. Use realistic values."],
            max_tokens=100
        )

        # Create batch config with json_mode enabled
        config = BatchConfig(
            json_mode=True,
            show_progress=False,
            print_prompts=False,
            print_responses=False
        )

        # Process batch with json_mode (sync)
        with ChatLimiter.for_model("gpt-3.5-turbo", api_key=api_key) as limiter:
            processor = ChatCompletionBatchProcessor(limiter, config)
            results = processor.process_batch_sync(requests)

        # Verify request succeeded
        assert len(results) == 1
        result = results[0]
        assert result.success, f"Request failed: {result.error_message}"
        assert result.result is not None
        assert len(result.result.choices) > 0
        
        # Extract the response content
        response_content = result.result.choices[0].message.content
        assert response_content is not None
        
        # Verify the response is valid JSON
        try:
            parsed_json = json.loads(response_content)
            assert isinstance(parsed_json, dict), "Response should be a JSON object"
            print(f"✅ Valid JSON response (sync): {parsed_json}")
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON: {response_content}. Error: {e}")

    def test_json_mode_disabled_by_default(self):
        """Test that json_mode is disabled by default."""
        config = BatchConfig()
        assert config.json_mode is False

    def test_json_mode_enabled_explicitly(self):
        """Test that json_mode can be enabled explicitly."""
        config = BatchConfig(json_mode=True)
        assert config.json_mode is True