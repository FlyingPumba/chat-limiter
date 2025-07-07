"""Tests for batch processing functionality."""

from unittest.mock import AsyncMock, Mock

import pytest

from chat_limiter import (
    BatchConfig,
    BatchItem,
    BatchResult,
    ChatBatchProcessor,
    ChatLimiter,
    Provider,
    process_chat_batch,
    process_chat_batch_sync,
)


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_batch_config_defaults(self):
        """Test BatchConfig default values."""
        config = BatchConfig()

        assert config.max_concurrent_requests == 10
        assert config.max_workers == 4
        assert config.max_retries_per_item == 3
        assert config.retry_delay == 1.0
        assert config.show_progress is True
        assert config.stop_on_first_error is False
        assert config.collect_errors is True
        assert config.verbose is False
        assert config.adaptive_batch_size is True
        assert config.group_by_model is True
        assert config.group_by_provider is True

    def test_batch_config_custom_values(self):
        """Test BatchConfig with custom values."""
        config = BatchConfig(
            max_concurrent_requests=20,
            max_retries_per_item=5,
            stop_on_first_error=True,
            group_by_model=False,
        )

        assert config.max_concurrent_requests == 20
        assert config.max_retries_per_item == 5
        assert config.stop_on_first_error is True
        assert config.group_by_model is False


class TestBatchItem:
    """Tests for BatchItem."""

    def test_batch_item_creation(self):
        """Test creating a BatchItem."""
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        item = BatchItem(data=data, id="test-1")

        assert item.data == data
        assert item.id == "test-1"
        assert item.method == "POST"
        assert item.url == "/chat/completions"
        assert item.json_data is None
        assert item.attempt_count == 0
        assert item.last_error is None
        assert item.metadata == {}

    def test_batch_item_with_custom_config(self):
        """Test BatchItem with custom configuration."""
        data = {"test": "data"}
        json_data = {"messages": [{"role": "user", "content": "Hello"}]}

        item = BatchItem(
            data=data,
            method="GET",
            url="/custom",
            json_data=json_data,
            id="custom-1",
            metadata={"custom": "meta"},
        )

        assert item.data == data
        assert item.method == "GET"
        assert item.url == "/custom"
        assert item.json_data == json_data
        assert item.id == "custom-1"
        assert item.metadata == {"custom": "meta"}


class TestBatchResult:
    """Tests for BatchResult."""

    def test_batch_result_success(self):
        """Test successful BatchResult."""
        item = BatchItem(data={"test": "data"}, id="test-1")
        result_data = {"response": "success"}

        result = BatchResult(
            item=item,
            result=result_data,
            success=True,
            duration=1.5,
            attempt_count=1,
            status_code=200,
        )

        assert result.item == item
        assert result.result == result_data
        assert result.success is True
        assert result.duration == 1.5
        assert result.attempt_count == 1
        assert result.status_code == 200
        assert result.error is None

    def test_batch_result_failure(self):
        """Test failed BatchResult."""
        item = BatchItem(data={"test": "data"}, id="test-1")
        error = Exception("Test error")

        result = BatchResult(
            item=item,
            error=error,
            success=False,
            duration=0.5,
            attempt_count=3,
            status_code=429,
        )

        assert result.item == item
        assert result.result is None
        assert result.success is False
        assert result.error == error
        assert result.duration == 0.5
        assert result.attempt_count == 3
        assert result.status_code == 429


class TestChatBatchProcessor:
    """Tests for ChatBatchProcessor."""

    @pytest.fixture
    def mock_limiter(self, mock_async_client, mock_sync_client):
        """Mock ChatLimiter for testing."""
        limiter = Mock(spec=ChatLimiter)
        limiter.provider = Provider.OPENAI
        limiter.request = AsyncMock()
        limiter.request_sync = Mock()
        return limiter

    @pytest.fixture
    def processor(self, mock_limiter):
        """ChatBatchProcessor instance for testing."""
        config = BatchConfig(max_concurrent_requests=2)
        return ChatBatchProcessor(mock_limiter, config)

    def test_create_batch_items(self, processor):
        """Test creating batch items from raw data."""
        raw_items = [
            {"message": "Hello 1"},
            {"message": "Hello 2"},
            {"message": "Hello 3"},
        ]

        batch_items = processor.create_batch_items(raw_items)

        assert len(batch_items) == 3
        for i, item in enumerate(batch_items):
            assert isinstance(item, BatchItem)
            assert item.data == raw_items[i]
            assert item.id == f"item_{i}"
            assert item.method == "POST"
            assert item.url == "/chat/completions"

    def test_create_batch_items_with_request_fn(self, processor):
        """Test creating batch items with request function."""
        raw_items = ["Hello 1", "Hello 2"]

        def request_fn(item):
            return "GET", f"/custom/{item}", {"data": item}

        batch_items = processor.create_batch_items(raw_items, request_fn)

        assert len(batch_items) == 2
        assert batch_items[0].method == "GET"
        assert batch_items[0].url == "/custom/Hello 1"
        assert batch_items[0].json_data == {"data": "Hello 1"}
        assert batch_items[1].method == "GET"
        assert batch_items[1].url == "/custom/Hello 2"
        assert batch_items[1].json_data == {"data": "Hello 2"}

    @pytest.mark.asyncio
    async def test_process_item_success(self, processor, mock_openai_response):
        """Test processing a single item successfully."""
        processor.limiter.request.return_value = mock_openai_response

        item = BatchItem(
            data={"test": "data"},
            json_data={"messages": [{"role": "user", "content": "Hello"}]},
        )

        result = await processor.process_item(item)

        assert result is not None
        processor.limiter.request.assert_called_once_with(
            method="POST",
            url="/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    def test_process_item_sync_success(self, processor, mock_openai_response):
        """Test processing a single item synchronously."""
        processor.limiter.request_sync.return_value = mock_openai_response

        item = BatchItem(
            data={"test": "data"},
            json_data={"messages": [{"role": "user", "content": "Hello"}]},
        )

        result = processor.process_item_sync(item)

        assert result is not None
        processor.limiter.request_sync.assert_called_once_with(
            method="POST",
            url="/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    @pytest.mark.asyncio
    async def test_process_batch_success(self, processor, mock_openai_response):
        """Test processing a batch successfully."""
        processor.limiter.request.return_value = mock_openai_response

        raw_items = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]

        results = await processor.process_batch(raw_items)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.success is True
            assert result.error is None

    def test_process_batch_sync_success(self, processor, mock_openai_response):
        """Test processing a batch synchronously."""
        processor.limiter.request_sync.return_value = mock_openai_response

        raw_items = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]

        results = processor.process_batch_sync(raw_items)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.success is True
            assert result.error is None

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, processor):
        """Test processing a batch with some errors."""
        # Mock first request to succeed, second to fail
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}
        success_response.headers = {}
        success_response.raise_for_status = Mock()

        def mock_request(*args, **kwargs):
            if "Hello 1" in str(kwargs.get("json", {})):
                return success_response
            else:
                raise Exception("Test error")

        processor.limiter.request.side_effect = mock_request

        raw_items = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]

        # Disable retries for faster test
        processor.config.max_retries_per_item = 0

        results = await processor.process_batch(raw_items)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error is not None

    def test_group_items_by_model(self, processor):
        """Test grouping items by model."""
        items = [
            BatchItem(data={}, json_data={"model": "gpt-3.5-turbo"}),
            BatchItem(data={}, json_data={"model": "gpt-4"}),
            BatchItem(data={}, json_data={"model": "gpt-3.5-turbo"}),
        ]

        groups = processor._group_items(items)

        assert "gpt-3.5-turbo" in groups
        assert "gpt-4" in groups
        assert len(groups["gpt-3.5-turbo"]) == 2
        assert len(groups["gpt-4"]) == 1

    def test_group_items_by_provider(self, processor):
        """Test grouping items by provider."""
        processor.config.group_by_model = False
        processor.config.group_by_provider = True

        items = [
            BatchItem(data={}),
            BatchItem(data={}),
        ]

        groups = processor._group_items(items)

        assert "openai" in groups
        assert len(groups["openai"]) == 2

    def test_get_stats(self, processor):
        """Test getting processing statistics."""
        # Create mock results
        successful_result = BatchResult(
            item=BatchItem(data={}),
            result={"success": True},
            success=True,
            duration=1.0,
            attempt_count=1,
        )
        failed_result = BatchResult(
            item=BatchItem(data={}),
            error=Exception("Test error"),
            success=False,
            duration=0.5,
            attempt_count=3,
        )

        processor._results = [successful_result, failed_result]

        stats = processor.get_stats()

        assert stats["total"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["avg_duration"] == 0.75
        assert stats["total_duration"] == 1.5
        assert stats["avg_attempts"] == 2.0

    def test_get_successful_results(self, processor):
        """Test getting only successful results."""
        successful_result = BatchResult(
            item=BatchItem(data={}),
            result={"success": True},
            success=True,
        )
        failed_result = BatchResult(
            item=BatchItem(data={}),
            error=Exception("Test error"),
            success=False,
        )

        processor._results = [successful_result, failed_result]

        successful = processor.get_successful_results()

        assert len(successful) == 1
        assert successful[0] == successful_result

    def test_get_failed_results(self, processor):
        """Test getting only failed results."""
        successful_result = BatchResult(
            item=BatchItem(data={}),
            result={"success": True},
            success=True,
        )
        failed_result = BatchResult(
            item=BatchItem(data={}),
            error=Exception("Test error"),
            success=False,
        )

        processor._results = [successful_result, failed_result]

        failed = processor.get_failed_results()

        assert len(failed) == 1
        assert failed[0] == failed_result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_chat_batch(self, mock_async_client, mock_openai_response):
        """Test process_chat_batch convenience function."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        mock_async_client.request.return_value = mock_openai_response

        requests = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]

        async with limiter:
            results = await process_chat_batch(limiter, requests)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.success is True

    def test_process_chat_batch_sync(self, mock_sync_client, mock_openai_response):
        """Test process_chat_batch_sync convenience function."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI,
            api_key="sk-test",
            sync_http_client=mock_sync_client,
        )

        mock_sync_client.request.return_value = mock_openai_response

        requests = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]

        with limiter:
            results = process_chat_batch_sync(limiter, requests)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, BatchResult)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_process_chat_batch_with_config(
        self, mock_async_client, mock_openai_response
    ):
        """Test process_chat_batch with custom config."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        mock_async_client.request.return_value = mock_openai_response

        requests = [{"messages": [{"role": "user", "content": "Hello"}]}]
        config = BatchConfig(max_concurrent_requests=1)

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_progress_bar_async(
        self, mock_async_client, mock_openai_response, capsys
    ):
        """Test that progress bar appears when show_progress=True."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        mock_async_client.request.return_value = mock_openai_response

        requests = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
            {"messages": [{"role": "user", "content": "Hello 3"}]},
        ]
        
        # Test with progress bar enabled
        config = BatchConfig(show_progress=True, progress_desc="Testing progress")

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        assert len(results) == 3
        for result in results:
            assert result.success is True

    def test_progress_bar_sync(
        self, mock_sync_client, mock_openai_response, capsys
    ):
        """Test that progress bar appears when show_progress=True in sync mode."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", sync_http_client=mock_sync_client
        )

        mock_sync_client.request.return_value = mock_openai_response

        requests = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
            {"messages": [{"role": "user", "content": "Hello 3"}]},
        ]
        
        # Test with progress bar enabled
        config = BatchConfig(show_progress=True, progress_desc="Testing sync progress")

        with limiter:
            results = process_chat_batch_sync(limiter, requests, config)

        assert len(results) == 3
        for result in results:
            assert result.success is True

    @pytest.mark.asyncio
    async def test_no_progress_bar_when_disabled(
        self, mock_async_client, mock_openai_response
    ):
        """Test that no progress bar appears when show_progress=False."""
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        mock_async_client.request.return_value = mock_openai_response

        requests = [
            {"messages": [{"role": "user", "content": "Hello 1"}]},
            {"messages": [{"role": "user", "content": "Hello 2"}]},
        ]
        
        # Test with progress bar disabled
        config = BatchConfig(show_progress=False)

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        assert len(results) == 2
        for result in results:
            assert result.success is True

    @pytest.mark.asyncio
    async def test_verbose_mode_prints_traceback(
        self, mock_async_client, capsys
    ):
        """Test that verbose=True prints tracebacks on exceptions."""
        # Create a mock that raises an exception
        mock_async_client.request.side_effect = Exception("Test exception")
        
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        requests = [{"messages": [{"role": "user", "content": "Hello"}]}]
        
        # Test with verbose mode enabled
        config = BatchConfig(
            verbose=True,
            show_progress=False,  # Disable progress bar for cleaner output
            max_retries_per_item=1  # Fail quickly for test
        )

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        # Check that the request failed as expected
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        
        # Check that traceback was printed to stdout and stderr
        captured = capsys.readouterr()
        assert "Exception in batch item" in captured.out
        assert "Traceback" in captured.err
        assert "Test exception" in captured.err

    def test_verbose_mode_sync_prints_traceback(
        self, mock_sync_client, capsys
    ):
        """Test that verbose=True prints tracebacks on exceptions in sync mode."""
        # Create a mock that raises an exception
        mock_sync_client.request.side_effect = Exception("Test sync exception")
        
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", sync_http_client=mock_sync_client
        )

        requests = [{"messages": [{"role": "user", "content": "Hello"}]}]
        
        # Test with verbose mode enabled
        config = BatchConfig(
            verbose=True,
            show_progress=False,  # Disable progress bar for cleaner output
            max_retries_per_item=1  # Fail quickly for test
        )

        with limiter:
            results = process_chat_batch_sync(limiter, requests, config)

        # Check that the request failed as expected
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        
        # Check that traceback was printed to stdout and stderr
        captured = capsys.readouterr()
        assert "Exception in batch item" in captured.out
        assert "Traceback" in captured.err
        assert "Test sync exception" in captured.err

    @pytest.mark.asyncio
    async def test_verbose_mode_disabled_no_traceback(
        self, mock_async_client, capsys
    ):
        """Test that verbose=False doesn't print tracebacks on exceptions."""
        # Create a mock that raises an exception
        mock_async_client.request.side_effect = Exception("Silent test exception")
        
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        requests = [{"messages": [{"role": "user", "content": "Hello"}]}]
        
        # Test with verbose mode disabled (default)
        config = BatchConfig(
            verbose=False,
            show_progress=False,  # Disable progress bar for cleaner output
            max_retries_per_item=1  # Fail quickly for test
        )

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        # Check that the request failed as expected
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        
        # Check that NO traceback was printed to stdout or stderr
        captured = capsys.readouterr()
        assert "Exception in batch item" not in captured.out
        assert "Traceback" not in captured.err
        assert "Silent test exception" not in captured.err

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, mock_async_client, capsys):
        """Test special handling for timeout errors in batch processing."""
        import httpx
        
        # Create a mock that raises a ReadTimeout exception
        mock_async_client.request.side_effect = httpx.ReadTimeout("Test timeout")
        
        limiter = ChatLimiter(
            provider=Provider.OPENAI, api_key="sk-test", http_client=mock_async_client
        )

        requests = [{"messages": [{"role": "user", "content": "Hello"}]}]
        
        # Test with verbose mode enabled to see timeout-specific messaging
        config = BatchConfig(
            verbose=True,
            show_progress=False,  # Disable progress bar for cleaner output
            max_retries_per_item=1,  # Fail quickly for test
            retry_delay=0.1  # Short delay for test
        )

        async with limiter:
            results = await process_chat_batch(limiter, requests, config)

        # Check that the request failed as expected
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        
        # Check that user-friendly timeout messaging was printed
        captured = capsys.readouterr()
        assert "⏱️  TIMEOUT ERROR in batch item" in captured.out
        assert "Current timeout setting:" in captured.out
        assert "💡 How to fix this:" in captured.out
        assert "1. Increase timeout: ChatLimiter.for_model" in captured.out
        assert "2. Reduce concurrency: BatchConfig" in captured.out
        assert "3. Current concurrency:" in captured.out
