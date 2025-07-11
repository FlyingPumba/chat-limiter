"""
Test error propagation from ChatCompletionResponse to BatchResult.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from chat_limiter.batch import ChatCompletionBatchProcessor, BatchItem
from chat_limiter.limiter import ChatLimiter
from chat_limiter.types import ChatCompletionRequest, ChatCompletionResponse, Message, MessageRole


class TestBatchErrorPropagation:
    """Test that errors from ChatCompletionResponse are properly propagated to BatchResult."""

    @pytest.fixture
    def mock_limiter(self):
        """Create a mock limiter for testing."""
        limiter = Mock(spec=ChatLimiter)
        limiter.chat_completion = AsyncMock()
        limiter.chat_completion_sync = Mock()
        # Add provider attribute needed for batch grouping
        limiter.provider = Mock()
        limiter.provider.value = "openai"
        return limiter

    @pytest.fixture
    def processor(self, mock_limiter):
        """Create a batch processor with mock limiter."""
        return ChatCompletionBatchProcessor(mock_limiter)

    @pytest.mark.asyncio
    async def test_chat_completion_error_propagation_async(self, processor, mock_limiter):
        """Test that ChatCompletionResponse errors are propagated to BatchResult in async processing."""
        # Create a ChatCompletionResponse with an error
        error_response = ChatCompletionResponse(
            id="test-id",
            model="gpt-3.5-turbo",
            choices=[],
            success=False,
            error_message="API authentication failed"
        )
        
        # Mock the limiter to return the error response
        mock_limiter.chat_completion.return_value = error_response
        
        # Create a batch item
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        items = [BatchItem(data=request, id="test-1")]
        
        # Process the batch
        results = await processor.process_batch(items)
        
        # Check that the error was propagated to BatchResult
        assert len(results) == 1
        result = results[0]
        
        assert result.success is False
        assert result.error_message is not None
        assert "Chat completion failed: API authentication failed" in result.error_message
        assert result.result is None

    def test_chat_completion_error_propagation_sync(self, processor, mock_limiter):
        """Test that ChatCompletionResponse errors are propagated to BatchResult in sync processing."""
        # Create a ChatCompletionResponse with an error
        error_response = ChatCompletionResponse(
            id="test-id",
            model="gpt-3.5-turbo",
            choices=[],
            success=False,
            error_message="Rate limit exceeded"
        )
        
        # Mock the limiter to return the error response
        mock_limiter.chat_completion_sync.return_value = error_response
        
        # Create a batch item
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        items = [BatchItem(data=request, id="test-1")]
        
        # Process the batch
        results = processor.process_batch_sync(items)
        
        # Check that the error was propagated to BatchResult
        assert len(results) == 1
        result = results[0]
        
        assert result.success is False
        assert result.error_message is not None
        assert "Chat completion failed: Rate limit exceeded" in result.error_message
        assert result.result is None

    @pytest.mark.asyncio
    async def test_successful_response_no_error_propagation(self, processor, mock_limiter):
        """Test that successful ChatCompletionResponse does not create errors in BatchResult."""
        # Create a successful ChatCompletionResponse
        success_response = ChatCompletionResponse(
            id="test-id",
            model="gpt-3.5-turbo",
            choices=[],
            success=True,
            error_message=None
        )
        
        # Mock the limiter to return the success response
        mock_limiter.chat_completion.return_value = success_response
        
        # Create a batch item
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello")]
        )
        items = [BatchItem(data=request, id="test-1")]
        
        # Process the batch
        results = await processor.process_batch(items)
        
        # Check that no error was created
        assert len(results) == 1
        result = results[0]
        
        assert result.success is True
        assert result.error_message is None
        assert result.result == success_response

    @pytest.mark.asyncio
    async def test_mixed_success_and_error_responses(self, processor, mock_limiter):
        """Test batch processing with both successful and error responses."""
        # Create responses
        success_response = ChatCompletionResponse(
            id="success-id",
            model="gpt-3.5-turbo",
            choices=[],
            success=True,
            error_message=None
        )
        
        error_response = ChatCompletionResponse(
            id="error-id",
            model="gpt-3.5-turbo",
            choices=[],
            success=False,
            error_message="Model not available"
        )
        
        # Mock the limiter to return different responses
        # We need to account for retries, so repeat the responses
        mock_limiter.chat_completion.side_effect = [
            success_response,  # First call succeeds
            error_response,    # Second call fails (first attempt)
            error_response,    # Second call fails (retry 1)
            error_response,    # Second call fails (retry 2)
            error_response     # Second call fails (retry 3)
        ]
        
        # Create batch items
        request1 = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello 1")]
        )
        request2 = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[Message(role=MessageRole.USER, content="Hello 2")]
        )
        items = [
            BatchItem(data=request1, id="test-1"),
            BatchItem(data=request2, id="test-2")
        ]
        
        # Process the batch
        results = await processor.process_batch(items)
        
        # Check results
        assert len(results) == 2
        
        # First result should be successful
        assert results[0].success is True
        assert results[0].error_message is None
        assert results[0].result == success_response
        
        # Second result should have error
        assert results[1].success is False
        assert results[1].error_message is not None
        assert "Chat completion failed: Model not available" in results[1].error_message
        assert results[1].result is None