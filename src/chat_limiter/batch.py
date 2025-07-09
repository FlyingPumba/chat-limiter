"""
Batch processing functionality for handling multiple requests efficiently.
"""

import asyncio
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

import httpx

from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

if TYPE_CHECKING:
    pass

from .limiter import ChatLimiter
from .types import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)

# Type variables for generic batch processing
BatchItemT = TypeVar("BatchItemT")
BatchResultT = TypeVar("BatchResultT")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    # Concurrency settings
    max_concurrent_requests: int = 10
    max_workers: int = 4  # For sync processing

    # Retry settings
    max_retries_per_item: int = 3
    retry_delay: float = 1.0

    # Progress tracking
    show_progress: bool = True
    progress_desc: str = "Processing batch"

    # Error handling
    stop_on_first_error: bool = False
    collect_errors: bool = True
    verbose: bool = False

    # Batch size optimization
    adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 100

    # Request grouping
    group_by_model: bool = True
    group_by_provider: bool = True


@dataclass
class BatchItem(Generic[BatchItemT]):
    """A single item in a batch request."""

    # Item data
    data: BatchItemT

    # Request configuration
    method: str = "POST"
    url: str = "/chat/completions"
    json_data: dict[str, Any] | None = None

    # Metadata
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Processing state
    attempt_count: int = 0
    last_error: Exception | None = None


@dataclass
class BatchResult(Generic[BatchResultT]):
    """Result of processing a batch item."""

    # Original item
    item: "BatchItem[Any]"

    # Result data
    result: BatchResultT | None = None
    error: Exception | None = None

    # Processing metadata
    success: bool = False
    duration: float = 0.0
    attempt_count: int = 0

    # Response metadata
    response_headers: dict[str, str] = field(default_factory=dict)
    status_code: int | None = None


class BatchProcessor(ABC, Generic[BatchItemT, BatchResultT]):
    """Abstract base class for batch processing."""

    def __init__(
        self,
        limiter: ChatLimiter,
        config: BatchConfig | None = None,
    ):
        self.limiter = limiter
        self.config = config or BatchConfig()
        self._results: list[BatchResult[BatchResultT]] = []
        self._errors: list[Exception] = []
        
        # Enable verbose mode on limiter if config specifies it
        if hasattr(self.limiter, 'set_verbose_mode'):
            self.limiter.set_verbose_mode(self.config.verbose)

    @abstractmethod
    async def process_item(self, item: BatchItem[BatchItemT]) -> BatchResultT:
        """Process a single batch item."""
        pass

    @abstractmethod
    def process_item_sync(self, item: BatchItem[BatchItemT]) -> BatchResultT:
        """Process a single batch item synchronously."""
        pass

    def create_batch_items(
        self,
        items: list[BatchItemT],
        request_fn: Callable[[BatchItemT], tuple[str, str, dict[str, Any]]] | None = None,
    ) -> list[BatchItem[BatchItemT]]:
        """Create batch items from raw data."""
        batch_items = []

        for i, item in enumerate(items):
            batch_item = BatchItem(
                data=item,
                id=f"item_{i}",
            )

            # Configure request if function provided
            if request_fn:
                method, url, json_data = request_fn(item)
                batch_item.method = method
                batch_item.url = url
                batch_item.json_data = json_data

            batch_items.append(batch_item)

        return batch_items

    async def process_batch(
        self,
        items: list[BatchItemT] | list[BatchItem[BatchItemT]],
        request_fn: Callable[[BatchItemT], tuple[str, str, dict[str, Any]]] | None = None,
    ) -> list[BatchResult[BatchResultT]]:
        """Process a batch of items asynchronously."""
        # Convert to batch items if needed
        if items and not isinstance(items[0], BatchItem):
            batch_items = self.create_batch_items(items, request_fn)  # type: ignore
        else:
            batch_items = items  # type: ignore

        # Group items if configured
        if self.config.group_by_model or self.config.group_by_provider:
            grouped_items = self._group_items(batch_items)
        else:
            grouped_items = {"default": batch_items}

        # Process groups
        all_results = []
        
        # Calculate total items for progress tracking
        total_items = sum(len(group_items) for group_items in grouped_items.values())
        
        # Initialize progress bar if enabled
        progress_bar = None
        if self.config.show_progress:
            progress_bar = tqdm(
                total=total_items,
                desc=self.config.progress_desc,
                unit="item"
            )
        
        for group_name, group_items in grouped_items.items():
            logger.info(
                f"Processing group '{group_name}' with {len(group_items)} items"
            )

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

            # Process items with concurrency control and progress tracking
            tasks = [
                self._process_item_with_retry(item, semaphore, progress_bar) for item in group_items
            ]

            # Wait for all tasks to complete
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions from gather
            for i, result in enumerate(group_results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result: BatchResult[BatchResultT] = BatchResult(
                        item=group_items[i],
                        error=result,
                        success=False,
                        attempt_count=group_items[i].attempt_count,
                    )
                    all_results.append(error_result)
                else:
                    all_results.append(result)  # type: ignore

        # Close progress bar if it was created
        if progress_bar:
            progress_bar.close()

        self._results = all_results
        return all_results

    def process_batch_sync(
        self,
        items: list[BatchItemT] | list[BatchItem[BatchItemT]],
        request_fn: Callable[[BatchItemT], tuple[str, str, dict[str, Any]]] | None = None,
    ) -> list[BatchResult[BatchResultT]]:
        """Process a batch of items synchronously."""
        # Convert to batch items if needed
        if items and not isinstance(items[0], BatchItem):
            batch_items = self.create_batch_items(items, request_fn)  # type: ignore
        else:
            batch_items = items  # type: ignore

        # Group items if configured
        if self.config.group_by_model or self.config.group_by_provider:
            grouped_items = self._group_items(batch_items)
        else:
            grouped_items = {"default": batch_items}

        # Calculate total items for progress tracking
        total_items = sum(len(group_items) for group_items in grouped_items.values())
        
        # Initialize progress bar if enabled
        progress_bar = None
        if self.config.show_progress:
            progress_bar = tqdm(
                total=total_items,
                desc=self.config.progress_desc,
                unit="item"
            )

        # Process groups
        all_results = []
        for group_name, group_items in grouped_items.items():
            logger.info(
                f"Processing group '{group_name}' with {len(group_items)} items"
            )

            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(self._process_item_sync_with_retry, item, progress_bar): item
                    for item in group_items
                }

                # Collect results
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        error_result: BatchResult[BatchResultT] = BatchResult(
                            item=item,
                            error=e,
                            success=False,
                            attempt_count=item.attempt_count,
                        )
                        all_results.append(error_result)

        # Close progress bar if it was created
        if progress_bar:
            progress_bar.close()

        self._results = all_results
        return all_results

    def _group_items(
        self, items: list[BatchItem[BatchItemT]]
    ) -> dict[str, list[BatchItem[BatchItemT]]]:
        """Group items by model or provider."""
        groups: dict[str, list[BatchItem[BatchItemT]]] = {}

        for item in items:
            # Determine group key
            group_key = "default"

            if (
                self.config.group_by_model
                and item.json_data
                and "model" in item.json_data
            ):
                group_key = item.json_data["model"]
            elif self.config.group_by_provider:
                group_key = self.limiter.provider.value

            # Add to group
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)

        return groups

    async def _process_item_with_retry(
        self,
        item: BatchItem[BatchItemT],
        semaphore: asyncio.Semaphore,
        progress_bar: tqdm | None = None,
    ) -> BatchResult[BatchResultT]:
        """Process a single item with retry logic."""
        async with semaphore:
            import time

            start_time = time.time()

            for attempt in range(self.config.max_retries_per_item + 1):
                item.attempt_count = attempt + 1

                try:
                    # Process the item
                    result = await self.process_item(item)

                    # Update progress bar on success
                    if progress_bar:
                        progress_bar.update(1)

                    # Success
                    return BatchResult(
                        item=item,
                        result=result,
                        success=True,
                        duration=time.time() - start_time,
                        attempt_count=item.attempt_count,
                    )

                except Exception as e:
                    item.last_error = e
                    
                    # Check if this is a timeout error
                    is_timeout_error = (
                        isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)) or
                        (hasattr(e, '__cause__') and isinstance(e.__cause__, (httpx.ReadTimeout, httpx.ConnectTimeout))) or
                        'ReadTimeout' in str(type(e)) or 'timeout' in str(e).lower()
                    )

                    # Print user-friendly error messages
                    if self.config.verbose:
                        if is_timeout_error:
                            # Get current timeout from the limiter
                            current_timeout = getattr(self.limiter, '_user_timeout', 120.0)
                            print(f"⏱️  TIMEOUT ERROR in batch item {item.id} (attempt {attempt + 1}):")
                            print(f"   Current timeout setting: {current_timeout} seconds")
                            print(f"   The request took longer than {current_timeout}s to complete.")
                            print(f"")
                            print(f"💡 How to fix this:")
                            print(f"   1. Increase timeout: ChatLimiter.for_model('{getattr(self.limiter, 'provider', 'your-model')}', timeout={current_timeout + 60})")
                            print(f"   2. Reduce concurrency: BatchConfig(max_concurrent_requests={max(1, self.config.max_concurrent_requests // 2)})")
                            print(f"   3. Current concurrency: {self.config.max_concurrent_requests} requests")
                            print(f"")
                        else:
                            print(f"❌ Exception in batch item {item.id} (attempt {attempt + 1}):")
                        
                        traceback.print_exc()

                    # If this is the last attempt or we should stop on error
                    if (
                        attempt == self.config.max_retries_per_item
                        or self.config.stop_on_first_error
                    ):
                        # Update progress bar on final failure
                        if progress_bar:
                            progress_bar.update(1)
                            
                        return BatchResult(
                            item=item,
                            error=e,
                            success=False,
                            duration=time.time() - start_time,
                            attempt_count=item.attempt_count,
                        )

                    # Wait before retry - longer for timeout errors
                    if is_timeout_error:
                        # For timeout errors, wait longer and suggest more aggressive backing off
                        retry_delay = self.config.retry_delay * (3**attempt)  # More aggressive backoff
                    else:
                        retry_delay = self.config.retry_delay * (2**attempt)
                    
                    await asyncio.sleep(retry_delay)

        # This should never be reached, but added for type checking
        return BatchResult(
            item=item,
            error=Exception("Unexpected error in retry logic"),
            success=False,
            duration=time.time() - start_time,
            attempt_count=item.attempt_count,
        )

    def _process_item_sync_with_retry(
        self,
        item: BatchItem[BatchItemT],
        progress_bar: tqdm | None = None,
    ) -> BatchResult[BatchResultT]:
        """Process a single item with retry logic (sync)."""
        import time

        start_time = time.time()

        for attempt in range(self.config.max_retries_per_item + 1):
            item.attempt_count = attempt + 1

            try:
                # Process the item
                result = self.process_item_sync(item)

                # Update progress bar on success
                if progress_bar:
                    progress_bar.update(1)

                # Success
                return BatchResult(
                    item=item,
                    result=result,
                    success=True,
                    duration=time.time() - start_time,
                    attempt_count=item.attempt_count,
                )

            except Exception as e:
                item.last_error = e

                # Print traceback if verbose mode is enabled
                if self.config.verbose:
                    print(f"Exception in batch item {item.id} (attempt {attempt + 1}):")
                    traceback.print_exc()

                # If this is the last attempt or we should stop on error
                if (
                    attempt == self.config.max_retries_per_item
                    or self.config.stop_on_first_error
                ):
                    # Update progress bar on final failure
                    if progress_bar:
                        progress_bar.update(1)
                        
                    return BatchResult(
                        item=item,
                        error=e,
                        success=False,
                        duration=time.time() - start_time,
                        attempt_count=item.attempt_count,
                    )

                # Wait before retry
                time.sleep(self.config.retry_delay * (2**attempt))

        # This should never be reached, but added for type checking
        return BatchResult(
            item=item,
            error=Exception("Unexpected error in retry logic"),
            success=False,
            duration=time.time() - start_time,
            attempt_count=item.attempt_count,
        )

    def get_success_rate(self) -> float:
        """Get the success rate of the last batch."""
        if not self._results:
            return 0.0

        successful = sum(1 for r in self._results if r.success)
        return successful / len(self._results)

    def get_successful_results(self) -> list[BatchResult[BatchResultT]]:
        """Get only successful results."""
        return [r for r in self._results if r.success]

    def get_failed_results(self) -> list[BatchResult[BatchResultT]]:
        """Get only failed results."""
        return [r for r in self._results if not r.success]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive processing statistics."""
        if not self._results:
            return {"total": 0, "successful": 0, "failed": 0, "success_rate": 0.0}

        successful = self.get_successful_results()
        failed = self.get_failed_results()

        # Calculate timing statistics
        durations = [r.duration for r in self._results]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total": len(self._results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": self.get_success_rate(),
            "avg_duration": avg_duration,
            "total_duration": sum(durations),
            "avg_attempts": sum(r.attempt_count for r in self._results)
            / len(self._results),
        }


class ChatBatchProcessor(BatchProcessor[dict[str, Any], dict[str, Any]]):
    """Batch processor for chat completion requests."""

    async def process_item(self, item: BatchItem[dict[str, Any]]) -> dict[str, Any]:
        """Process a single chat completion request."""
        request_data = item.json_data or item.data
        
        # Log prompt if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- PROMPT (Item {item.id}) ---")
            if "messages" in request_data:
                for msg in request_data["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    print(f"{role.upper()}: {content}")
            else:
                print(f"REQUEST DATA: {request_data}")
            print("--- END PROMPT ---\n")
        
        # Make the request using the limiter
        response = await self.limiter.request(
            method=item.method,
            url=item.url,
            json=request_data,
        )

        # Parse response
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        
        # Log response if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- RESPONSE (Item {item.id}) ---")
            if "choices" in result and result["choices"]:
                for i, choice in enumerate(result["choices"]):
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                        print(f"CHOICE {i}: {content}")
                    elif "text" in choice:
                        print(f"CHOICE {i}: {choice['text']}")
            else:
                print(f"FULL RESPONSE: {result}")
            print("--- END RESPONSE ---\n")

        # Store response metadata
        item.metadata["response_headers"] = dict(response.headers)
        item.metadata["status_code"] = response.status_code

        return result

    def process_item_sync(self, item: BatchItem[dict[str, Any]]) -> dict[str, Any]:
        """Process a single chat completion request synchronously."""
        request_data = item.json_data or item.data
        
        # Log prompt if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- PROMPT (Item {item.id}) ---")
            if "messages" in request_data:
                for msg in request_data["messages"]:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    print(f"{role.upper()}: {content}")
            else:
                print(f"REQUEST DATA: {request_data}")
            print("--- END PROMPT ---\n")
        
        # Make the request using the limiter
        response = self.limiter.request_sync(
            method=item.method,
            url=item.url,
            json=request_data,
        )

        # Parse response
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        
        # Log response if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- RESPONSE (Item {item.id}) ---")
            if "choices" in result and result["choices"]:
                for i, choice in enumerate(result["choices"]):
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                        print(f"CHOICE {i}: {content}")
                    elif "text" in choice:
                        print(f"CHOICE {i}: {choice['text']}")
            else:
                print(f"FULL RESPONSE: {result}")
            print("--- END RESPONSE ---\n")

        # Store response metadata
        item.metadata["response_headers"] = dict(response.headers)
        item.metadata["status_code"] = response.status_code

        return result


# Convenience functions for common use cases
async def process_chat_batch(
    limiter: ChatLimiter,
    requests: list[dict[str, Any]],
    config: BatchConfig | None = None,
) -> list[BatchResult[dict[str, Any]]]:
    """
    Process a batch of chat completion requests.

    Args:
        limiter: Configured ChatLimiter instance
        requests: List of request data (will be sent as JSON)
        config: Optional batch processing configuration

    Returns:
        List of batch results
    """
    processor = ChatBatchProcessor(limiter, config)
    return await processor.process_batch(requests)


def process_chat_batch_sync(
    limiter: ChatLimiter,
    requests: list[dict[str, Any]],
    config: BatchConfig | None = None,
) -> list[BatchResult[dict[str, Any]]]:
    """
    Process a batch of chat completion requests synchronously.

    Args:
        limiter: Configured ChatLimiter instance
        requests: List of request data (will be sent as JSON)
        config: Optional batch processing configuration

    Returns:
        List of batch results
    """
    processor = ChatBatchProcessor(limiter, config)
    return processor.process_batch_sync(requests)


# High-level chat completion batch processing
class ChatCompletionBatchProcessor(BatchProcessor[ChatCompletionRequest, ChatCompletionResponse]):
    """High-level batch processor for chat completion requests."""

    async def process_item(self, item: BatchItem[ChatCompletionRequest]) -> ChatCompletionResponse:
        """Process a single chat completion request using high-level interface."""
        request = item.data

        # Log prompt if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- PROMPT (Item {item.id}) ---")
            print(f"MODEL: {request.model}")
            for msg in request.messages:
                print(f"{msg.role.value.upper()}: {msg.content}")
            print("--- END PROMPT ---\n")

        # Use the high-level chat completion method
        response = await self.limiter.chat_completion(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stream=request.stream,
            # Provider-specific parameters
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_k=request.top_k,
        )

        # Log response if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- RESPONSE (Item {item.id}) ---")
            print(f"MODEL: {response.model}")
            if response.choices:
                for i, choice in enumerate(response.choices):
                    print(f"CHOICE {i}: {choice.message.content}")
            print("--- END RESPONSE ---\n")

        return response

    def process_item_sync(self, item: BatchItem[ChatCompletionRequest]) -> ChatCompletionResponse:
        """Process a single chat completion request synchronously using high-level interface."""
        request = item.data

        # Log prompt if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- PROMPT (Item {item.id}) ---")
            print(f"MODEL: {request.model}")
            for msg in request.messages:
                print(f"{msg.role.value.upper()}: {msg.content}")
            print("--- END PROMPT ---\n")

        # Use the high-level chat completion method (sync)
        response = self.limiter.chat_completion_sync(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stream=request.stream,
            # Provider-specific parameters
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_k=request.top_k,
        )

        # Log response if verbose mode is enabled
        if self.config.verbose:
            print(f"\n--- RESPONSE (Item {item.id}) ---")
            print(f"MODEL: {response.model}")
            if response.choices:
                for i, choice in enumerate(response.choices):
                    print(f"CHOICE {i}: {choice.message.content}")
            print("--- END RESPONSE ---\n")

        return response


# Convenience functions for high-level chat completion batches
async def process_chat_completion_batch(
    limiter: ChatLimiter,
    requests: list[ChatCompletionRequest],
    config: BatchConfig | None = None,
) -> list[BatchResult[ChatCompletionResponse]]:
    """
    Process a batch of high-level chat completion requests.

    Args:
        limiter: Configured ChatLimiter instance
        requests: List of ChatCompletionRequest objects
        config: Optional batch processing configuration

    Returns:
        List of batch results containing ChatCompletionResponse objects

    Example:
        from chat_limiter import ChatLimiter, Message, MessageRole, ChatCompletionRequest

        requests = [
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content="Hello!")],
                max_tokens=50
            ),
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content="How are you?")],
                max_tokens=50
            )
        ]

        async with ChatLimiter.for_model("gpt-4o", api_key) as limiter:
            results = await process_chat_completion_batch(limiter, requests)
    """
    processor = ChatCompletionBatchProcessor(limiter, config)
    return await processor.process_batch(requests)


def process_chat_completion_batch_sync(
    limiter: ChatLimiter,
    requests: list[ChatCompletionRequest],
    config: BatchConfig | None = None,
) -> list[BatchResult[ChatCompletionResponse]]:
    """
    Process a batch of high-level chat completion requests synchronously.

    Args:
        limiter: Configured ChatLimiter instance
        requests: List of ChatCompletionRequest objects
        config: Optional batch processing configuration

    Returns:
        List of batch results containing ChatCompletionResponse objects
    """
    processor = ChatCompletionBatchProcessor(limiter, config)
    return processor.process_batch_sync(requests)


# Helper function for creating chat completion requests from simple data
def create_chat_completion_requests(
    model: str,
    prompts: list[str],
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> list[ChatCompletionRequest]:
    """
    Create a list of ChatCompletionRequest objects from simple prompts.

    Args:
        model: The model to use for all requests
        prompts: List of user prompts
        max_tokens: Maximum tokens per completion
        temperature: Sampling temperature
        **kwargs: Additional parameters for all requests

    Returns:
        List of ChatCompletionRequest objects

    Example:
        requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=["Hello!", "How are you?", "What is Python?"],
            max_tokens=50,
            temperature=0.7
        )
    """
    from .types import Message, MessageRole

    requests = []
    for prompt in prompts:
        request = ChatCompletionRequest(
            model=model,
            messages=[Message(role=MessageRole.USER, content=prompt)],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        requests.append(request)

    return requests
