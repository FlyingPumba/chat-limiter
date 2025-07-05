"""
chat-limiter: A Pythonic rate limiter for OpenAI, Anthropic, and OpenRouter APIs
"""

__version__ = "0.1.0"
__author__ = "Ivan Arcuschin"
__email__ = "ivan@arcuschin.com"

from .batch import (
    BatchConfig,
    BatchItem,
    BatchProcessor,
    BatchResult,
    ChatBatchProcessor,
    ChatCompletionBatchProcessor,
    create_chat_completion_requests,
    process_chat_batch,
    process_chat_batch_sync,
    process_chat_completion_batch,
    process_chat_completion_batch_sync,
)
from .limiter import ChatLimiter, LimiterState
from .providers import Provider, ProviderConfig, RateLimitInfo
from .types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    MessageRole,
    Usage,
)

__all__ = [
    "ChatLimiter",
    "LimiterState",
    "Provider",
    "ProviderConfig",
    "RateLimitInfo",
    "BatchConfig",
    "BatchItem",
    "BatchResult",
    "BatchProcessor",
    "ChatBatchProcessor",
    "process_chat_batch",
    "process_chat_batch_sync",
    "ChatCompletionBatchProcessor",
    "process_chat_completion_batch",
    "process_chat_completion_batch_sync",
    "create_chat_completion_requests",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "MessageRole",
    "Usage",
    "Choice",
]
