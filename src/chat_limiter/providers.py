"""
Provider-specific configurations and rate limit header mappings.
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class Provider(Enum):
    """Supported API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


@dataclass
class RateLimitInfo:
    """Information about current rate limits."""

    # Request limits
    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: int | float | None = None  # Unix timestamp or seconds

    # Token limits
    tokens_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_reset: int | float | None = None  # Unix timestamp or seconds

    # Retry information
    retry_after: float | None = None  # Seconds to wait

    # Provider-specific metadata
    metadata: dict[str, str] = field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for API provider rate limit handling."""

    provider: Provider
    base_url: str

    # Rate limit header mappings
    request_limit_header: str | None = None
    request_remaining_header: str | None = None
    request_reset_header: str | None = None

    token_limit_header: str | None = None
    token_remaining_header: str | None = None
    token_reset_header: str | None = None

    retry_after_header: str | None = None

    # Rate limits (must be provided by user or discovered from API)
    default_request_limit: int | None = None
    default_token_limit: int | None = None

    # Rate limit discovery
    supports_dynamic_limits: bool = True
    auth_endpoint: str | None = None  # For checking limits via API

    # Retry configuration (no defaults - must be specified by user if needed)
    max_retries: int | None = None
    base_backoff: float | None = None
    max_backoff: float = Field(default=60.0, ge=1.0)

    # Safety buffers
    request_buffer_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    token_buffer_ratio: float = Field(default=0.9, ge=0.1, le=1.0)


# Provider-specific configurations
PROVIDER_CONFIGS = {
    Provider.OPENAI: ProviderConfig(
        provider=Provider.OPENAI,
        base_url="https://api.openai.com/v1",
        request_limit_header="x-ratelimit-limit-requests",
        request_remaining_header="x-ratelimit-remaining-requests",
        request_reset_header="x-ratelimit-reset-requests",
        token_limit_header="x-ratelimit-limit-tokens",
        token_remaining_header="x-ratelimit-remaining-tokens",
        token_reset_header="x-ratelimit-reset-tokens",
        retry_after_header="retry-after",
        supports_dynamic_limits=True,
    ),
    Provider.ANTHROPIC: ProviderConfig(
        provider=Provider.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        request_remaining_header="anthropic-ratelimit-requests-remaining",
        token_limit_header="anthropic-ratelimit-tokens-limit",
        token_reset_header="anthropic-ratelimit-tokens-reset",
        retry_after_header="retry-after",
        supports_dynamic_limits=True,
    ),
    Provider.OPENROUTER: ProviderConfig(
        provider=Provider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        auth_endpoint="https://openrouter.ai/api/v1/auth/key",
        supports_dynamic_limits=True,
    ),
}


def get_provider_config(provider: Provider) -> ProviderConfig:
    """Get configuration for a specific provider."""
    return PROVIDER_CONFIGS[provider]


def detect_provider_from_url(url: str) -> Provider | None:
    """Detect provider from API URL."""
    url_lower = url.lower()

    if "openai.com" in url_lower:
        return Provider.OPENAI
    elif "anthropic.com" in url_lower:
        return Provider.ANTHROPIC
    elif "openrouter.ai" in url_lower:
        return Provider.OPENROUTER

    return None


def extract_rate_limit_info(
    headers: dict[str, str], config: ProviderConfig
) -> RateLimitInfo:
    """Extract rate limit information from response headers."""
    info = RateLimitInfo()

    # Extract request limits
    if config.request_limit_header:
        info.requests_limit = _safe_int(headers.get(config.request_limit_header))
    if config.request_remaining_header:
        info.requests_remaining = _safe_int(
            headers.get(config.request_remaining_header)
        )
    if config.request_reset_header:
        info.requests_reset = _safe_float(headers.get(config.request_reset_header))

    # Extract token limits
    if config.token_limit_header:
        info.tokens_limit = _safe_int(headers.get(config.token_limit_header))
    if config.token_remaining_header:
        info.tokens_remaining = _safe_int(headers.get(config.token_remaining_header))
    if config.token_reset_header:
        info.tokens_reset = _safe_float(headers.get(config.token_reset_header))

    # Extract retry information
    if config.retry_after_header:
        info.retry_after = _safe_float(headers.get(config.retry_after_header))

    # Store all headers as metadata
    info.metadata = dict(headers)

    return info


def _safe_int(value: str | None) -> int | None:
    """Safely convert string to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: str | None) -> float | None:
    """Safely convert string to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
