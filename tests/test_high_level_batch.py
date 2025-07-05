"""
Tests for the high-level batch processing interface.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from chat_limiter import (
    BatchConfig,
    ChatCompletionBatchProcessor,
    ChatCompletionRequest,
    ChatLimiter,
    Message,
    MessageRole,
    Provider,
    create_chat_completion_requests,
    process_chat_completion_batch,
    process_chat_completion_batch_sync,
)


class TestCreateChatCompletionRequests:
    def test_create_requests_basic(self):
        """Test creating requests from simple prompts."""
        prompts = ["Hello!", "How are you?", "Goodbye!"]
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=prompts
        )

        assert len(requests) == 3
        assert all(req.model == "gpt-4o" for req in requests)
        assert requests[0].messages[0].content == "Hello!"
        assert requests[1].messages[0].content == "How are you?"
        assert requests[2].messages[0].content == "Goodbye!"

        # Check that all messages are user messages
        for req in requests:
            assert len(req.messages) == 1
            assert req.messages[0].role == MessageRole.USER

    def test_create_requests_with_parameters(self):
        """Test creating requests with additional parameters."""
        prompts = ["Test prompt"]
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=prompts,
            max_tokens=100,
            temperature=0.7,
            frequency_penalty=0.5
        )

        request = requests[0]
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.frequency_penalty == 0.5

    def test_create_requests_empty_list(self):
        """Test creating requests from empty prompt list."""
        requests = create_chat_completion_requests("gpt-4o", [])
        assert len(requests) == 0


class TestChatCompletionBatchProcessor:
    @pytest.fixture
    def mock_limiter(self):
        """Create a mock ChatLimiter for testing."""
        limiter = Mock(spec=ChatLimiter)

        # Mock the provider attribute
        limiter.provider = Provider.OPENAI

        # Mock the chat completion methods
        async def mock_chat_completion(**kwargs):
            from chat_limiter.types import ChatCompletionResponse, Choice, Usage
            return ChatCompletionResponse(
                id="test-response",
                model=kwargs["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content="Test response"),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                provider="openai"
            )

        def mock_chat_completion_sync(**kwargs):
            from chat_limiter.types import ChatCompletionResponse, Choice, Usage
            return ChatCompletionResponse(
                id="test-response",
                model=kwargs["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content="Test response"),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                provider="openai"
            )

        limiter.chat_completion = AsyncMock(side_effect=mock_chat_completion)
        limiter.chat_completion_sync = Mock(side_effect=mock_chat_completion_sync)

        return limiter

    @pytest.mark.asyncio
    async def test_process_item_async(self, mock_limiter):
        """Test processing a single item asynchronously."""
        processor = ChatCompletionBatchProcessor(mock_limiter)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        from chat_limiter.batch import BatchItem
        item = BatchItem(data=request)

        response = await processor.process_item(item)

        assert response.model == "gpt-4o"
        assert response.choices[0].message.content == "Test response"
        mock_limiter.chat_completion.assert_called_once()

    def test_process_item_sync(self, mock_limiter):
        """Test processing a single item synchronously."""
        processor = ChatCompletionBatchProcessor(mock_limiter)

        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )

        from chat_limiter.batch import BatchItem
        item = BatchItem(data=request)

        response = processor.process_item_sync(item)

        assert response.model == "gpt-4o"
        assert response.choices[0].message.content == "Test response"
        mock_limiter.chat_completion_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_async(self, mock_limiter):
        """Test processing a batch of requests asynchronously."""
        processor = ChatCompletionBatchProcessor(mock_limiter)

        requests = [
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content=f"Hello {i}!")]
            )
            for i in range(3)
        ]

        results = await processor.process_batch(requests)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.result is not None for result in results)

        # Check that chat_completion was called for each request
        assert mock_limiter.chat_completion.call_count == 3

    def test_process_batch_sync(self, mock_limiter):
        """Test processing a batch of requests synchronously."""
        processor = ChatCompletionBatchProcessor(mock_limiter)

        requests = [
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content=f"Hello {i}!")]
            )
            for i in range(3)
        ]

        results = processor.process_batch_sync(requests)

        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.result is not None for result in results)

        # Check that chat_completion_sync was called for each request
        assert mock_limiter.chat_completion_sync.call_count == 3


class TestConvenienceFunctions:
    @pytest.fixture
    def mock_limiter(self):
        """Create a mock ChatLimiter for testing."""
        limiter = Mock(spec=ChatLimiter)

        # Mock the provider attribute
        limiter.provider = Provider.OPENAI

        # Mock the chat completion methods
        async def mock_chat_completion(**kwargs):
            from chat_limiter.types import ChatCompletionResponse, Choice, Usage
            return ChatCompletionResponse(
                id="test-response",
                model=kwargs["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content=f"Response to: {kwargs['messages'][0].content}"),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                provider="openai"
            )

        def mock_chat_completion_sync(**kwargs):
            from chat_limiter.types import ChatCompletionResponse, Choice, Usage
            return ChatCompletionResponse(
                id="test-response",
                model=kwargs["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content=f"Response to: {kwargs['messages'][0].content}"),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                provider="openai"
            )

        limiter.chat_completion = AsyncMock(side_effect=mock_chat_completion)
        limiter.chat_completion_sync = Mock(side_effect=mock_chat_completion_sync)

        return limiter

    @pytest.mark.asyncio
    async def test_process_chat_completion_batch_async(self, mock_limiter):
        """Test the async convenience function."""
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Hello!", "How are you?"]
        )

        results = await process_chat_completion_batch(mock_limiter, requests)

        assert len(results) == 2
        assert all(result.success for result in results)

        # Check the responses
        assert "Response to: Hello!" in results[0].result.choices[0].message.content
        assert "Response to: How are you?" in results[1].result.choices[0].message.content

    def test_process_chat_completion_batch_sync(self, mock_limiter):
        """Test the sync convenience function."""
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Hello!", "How are you?"]
        )

        results = process_chat_completion_batch_sync(mock_limiter, requests)

        assert len(results) == 2
        assert all(result.success for result in results)

        # Check the responses
        assert "Response to: Hello!" in results[0].result.choices[0].message.content
        assert "Response to: How are you?" in results[1].result.choices[0].message.content

    @pytest.mark.asyncio
    async def test_process_chat_completion_batch_with_config(self, mock_limiter):
        """Test batch processing with custom configuration."""
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Test 1", "Test 2", "Test 3"]
        )

        config = BatchConfig(
            max_concurrent_requests=2,
            max_retries_per_item=1,
            group_by_model=True
        )

        results = await process_chat_completion_batch(mock_limiter, requests, config)

        assert len(results) == 3
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_process_chat_completion_batch_with_errors(self, mock_limiter):
        """Test batch processing with some failures."""
        # Make the second call fail
        async def mock_chat_completion_with_error(**kwargs):
            if "Test 2" in kwargs['messages'][0].content:
                raise Exception("Simulated error")

            from chat_limiter.types import ChatCompletionResponse, Choice, Usage
            return ChatCompletionResponse(
                id="test-response",
                model=kwargs["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role=MessageRole.ASSISTANT, content="Success"),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                provider="openai"
            )

        mock_limiter.chat_completion = AsyncMock(side_effect=mock_chat_completion_with_error)

        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Test 1", "Test 2", "Test 3"]
        )

        config = BatchConfig(
            max_retries_per_item=1,
            stop_on_first_error=False
        )

        results = await process_chat_completion_batch(mock_limiter, requests, config)

        assert len(results) == 3

        # Check that we have successes and failures
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 2
        assert len(failed) == 1

        # Check that the failed result has the error
        assert failed[0].error is not None


class TestBatchProcessingIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_batch_processing(self):
        """Test end-to-end batch processing with mock HTTP client."""
        # Create mock HTTP response
        def create_mock_response(content):
            mock_response = Mock()
            mock_response.json.return_value = {
                "id": "chatcmpl-test",
                "model": "gpt-4o-2024-08-06",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
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
            # Mock headers as a dict-like object
            mock_response.headers = {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "99",
            }
            return mock_response

        # Create mock HTTP client that returns different responses
        mock_client = AsyncMock()

        def mock_request_side_effect(*args, **kwargs):
            json_data = kwargs.get("json", {})
            messages = json_data.get("messages", [])
            if messages:
                user_content = messages[0].get("content", "")
                return create_mock_response(f"Response to: {user_content}")
            return create_mock_response("Default response")

        mock_client.request.side_effect = mock_request_side_effect
        mock_client.aclose = AsyncMock()

        # Create limiter with mock client
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test-key",
            http_client=mock_client
        )

        # Create requests
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Hello!", "How are you?"],
            max_tokens=50
        )

        # Process batch
        async with limiter:
            results = await process_chat_completion_batch(limiter, requests)

        # Verify results
        assert len(results) == 2
        assert all(result.success for result in results)

        # Check that responses match the prompts
        response_contents = [result.result.choices[0].message.content for result in results if result.result]
        assert "Response to: Hello!" in response_contents[0]
        assert "Response to: How are you?" in response_contents[1]

        # Verify that the HTTP client was called correctly
        assert mock_client.request.call_count == 2
