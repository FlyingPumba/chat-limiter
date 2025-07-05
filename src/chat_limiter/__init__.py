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
    process_chat_batch,
    process_chat_batch_sync,
)
from .limiter import ChatLimiter, LimiterState
from .providers import Provider, ProviderConfig, RateLimitInfo

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
]
