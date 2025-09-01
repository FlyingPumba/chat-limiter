"""Tests for batch processing functionality."""

from unittest.mock import AsyncMock, Mock

import pytest

from chat_limiter import (
    BatchConfig,
    BatchItem,
    BatchResult,
    ChatLimiter,
    Provider,
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
        # Test fine-grained logging configurations
        assert config.print_prompts is False
        assert config.print_responses is False
        assert config.verbose_timeouts is False
        assert config.verbose_exceptions is False
        assert config.print_rate_limits is False
        assert config.print_request_initiation is False
        
        # Test response format configuration
        assert config.json_mode is False
        assert config.adaptive_batch_size is True
        assert config.group_by_model is True
        assert config.group_by_provider is True
        
        # Test reasoning configuration
        assert config.reasoning_effort is None

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

    def test_reasoning_effort_validation(self):
        """Test reasoning_effort validation in BatchConfig."""
        # Valid values should work
        for effort in [None, "low", "medium", "high"]:
            config = BatchConfig(reasoning_effort=effort)
            assert config.reasoning_effort == effort

        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            BatchConfig(reasoning_effort="invalid")

        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            BatchConfig(reasoning_effort="extreme")

        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            BatchConfig(reasoning_effort="")


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
        assert result.error_message is None

    def test_batch_result_failure(self):
        """Test failed BatchResult."""
        item = BatchItem(data={"test": "data"}, id="test-1")
        error = Exception("Test error")

        result = BatchResult(
            item=item,
            success=False,
            error_message=str(error),
            duration=0.5,
            attempt_count=3,
            status_code=429,
        )

        assert result.item == item
        assert result.result is None
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.duration == 0.5
        assert result.attempt_count == 3
        assert result.status_code == 429