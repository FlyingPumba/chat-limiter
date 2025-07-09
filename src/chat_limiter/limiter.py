"""
Core rate limiter implementation using PyrateLimiter.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from pyrate_limiter import Duration, Limiter, Rate
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .adapters import get_adapter
from .providers import (
    Provider,
    ProviderConfig,
    RateLimitInfo,
    detect_provider_from_url,
    extract_rate_limit_info,
    get_provider_config,
)
from .types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    MessageRole,
    detect_provider_from_model,
)

logger = logging.getLogger(__name__)


@dataclass
class LimiterState:
    """Current state of the rate limiter."""

    # Current limits (None if not yet discovered)
    request_limit: int | None = None
    token_limit: int | None = None

    # Usage tracking
    requests_used: int = 0
    tokens_used: int = 0

    # Timing
    last_request_time: float = field(default_factory=time.time)
    last_limit_update: float = field(default_factory=time.time)

    # Rate limit info from last response
    last_rate_limit_info: RateLimitInfo | None = None

    # Adaptive behavior
    consecutive_rate_limit_errors: int = 0
    adaptive_backoff_factor: float = 1.0


class ChatLimiter:
    """
    A Pythonic rate limiter for API calls supporting OpenAI, Anthropic, and OpenRouter.

    Features:
    - Automatic rate limit discovery and adaptation
    - Sync and async support with context managers
    - Intelligent retry logic with exponential backoff
    - Token and request rate limiting
    - Provider-specific optimizations

    Example:
        # High-level interface (recommended)
        async with ChatLimiter.for_model("gpt-4o", api_key="sk-...") as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content="Hello!")]
            )

        # Low-level interface (for advanced users)
        async with ChatLimiter(provider=Provider.OPENAI, api_key="sk-...") as limiter:
            response = await limiter.request("POST", "/chat/completions", json=data)
    """

    def __init__(
        self,
        provider: Provider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        config: ProviderConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        sync_http_client: httpx.Client | None = None,
        enable_adaptive_limits: bool = True,
        enable_token_estimation: bool = True,
        request_limit: int | None = None,
        token_limit: int | None = None,
        max_retries: int | None = None,
        base_backoff: float | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the ChatLimiter.

        Args:
            provider: The API provider (OpenAI, Anthropic, OpenRouter)
            api_key: API key for authentication
            base_url: Base URL for API requests
            config: Custom provider configuration
            http_client: Custom async HTTP client
            sync_http_client: Custom sync HTTP client
            enable_adaptive_limits: Enable adaptive rate limit adjustment
            enable_token_estimation: Enable token usage estimation
            request_limit: Override request limit (if not provided, must be discovered from API)
            token_limit: Override token limit (if not provided, must be discovered from API)
            max_retries: Override max retries (defaults to 3 if not provided)
            base_backoff: Override base backoff (defaults to 1.0 if not provided)
            timeout: HTTP request timeout in seconds (defaults to 120.0 for better reliability)
            **kwargs: Additional arguments passed to HTTP clients
        """
        # Determine provider and config
        if config:
            self.config = config
            self.provider = config.provider
        elif provider:
            self.provider = provider
            self.config = get_provider_config(provider)
        elif base_url:
            detected_provider = detect_provider_from_url(base_url)
            if detected_provider:
                self.provider = detected_provider
                self.config = get_provider_config(detected_provider)
            else:
                raise ValueError(f"Could not detect provider from URL: {base_url}")
        else:
            raise ValueError("Must provide either provider, config, or base_url")

        # Override base_url if provided
        if base_url:
            self.config.base_url = base_url

        # Store configuration
        self.api_key = api_key
        self.enable_adaptive_limits = enable_adaptive_limits
        self.enable_token_estimation = enable_token_estimation
        
        # Store user-provided overrides
        self._user_request_limit = request_limit
        self._user_token_limit = token_limit
        self._user_max_retries = max_retries or 3  # Default to 3 if not provided
        self._user_base_backoff = base_backoff or 1.0  # Default to 1.0 if not provided
        self._user_timeout = timeout or 120.0  # Default to 120 seconds for better reliability

        # Determine initial limits (user override, config default, or None for discovery)
        initial_request_limit = (
            request_limit or 
            self.config.default_request_limit or 
            None
        )
        initial_token_limit = (
            token_limit or 
            self.config.default_token_limit or 
            None
        )
        
        # Initialize state - will be None if no defaults and no discovery yet
        self.state = LimiterState(
            request_limit=initial_request_limit,
            token_limit=initial_token_limit,
        )
        
        # Flag to track if we need to discover limits
        self._limits_discovered = initial_request_limit is not None and initial_token_limit is not None

        # Initialize HTTP clients
        self._init_http_clients(http_client, sync_http_client, **kwargs)

        # Initialize rate limiters
        self._init_rate_limiters()

        # Context manager state
        self._async_context_active = False
        self._sync_context_active = False
        
        # Verbose mode (can be set by batch processor)
        self._verbose_mode = False

    @classmethod
    def for_model(
        cls,
        model: str,
        api_key: str | None = None,
        provider: str | Provider | None = None,
        use_dynamic_discovery: bool = True,
        request_limit: int | None = None,
        token_limit: int | None = None,
        max_retries: int | None = None,
        base_backoff: float | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> "ChatLimiter":
        """
        Create a ChatLimiter instance automatically detecting the provider from the model name.

        Args:
            model: The model name (e.g., "gpt-4o", "claude-3-sonnet-20240229")
            api_key: API key for the provider. If None, will be read from environment variables
                    (OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY)
            provider: Override provider detection. Can be "openai", "anthropic", "openrouter",
                     or Provider enum. If None, will be auto-detected from model name
            use_dynamic_discovery: Whether to query live APIs for model availability (default: True).
                                 Requires appropriate API keys to be available. Falls back to
                                 hardcoded model lists when disabled or when API calls fail.
            **kwargs: Additional arguments passed to ChatLimiter

        Returns:
            Configured ChatLimiter instance

        Raises:
            ValueError: If provider cannot be determined from model name or API key not found

        Example:
            # Auto-detect provider with dynamic discovery (default behavior)
            async with ChatLimiter.for_model("gpt-4o") as limiter:
                response = await limiter.simple_chat("gpt-4o", "Hello!")

            # Override provider detection
            async with ChatLimiter.for_model("custom-model", provider="openai") as limiter:
                response = await limiter.simple_chat("custom-model", "Hello!")

            # Disable dynamic discovery to use only hardcoded model lists
            async with ChatLimiter.for_model("gpt-4o", use_dynamic_discovery=False) as limiter:
                response = await limiter.simple_chat("gpt-4o", "Hello!")
        """
        import os

        # Determine provider
        if provider is not None:
            # Use provided provider
            if isinstance(provider, str):
                provider_enum = Provider(provider)
            else:
                provider_enum = provider
            provider_name = provider_enum.value
        else:
            # Auto-detect from model name
            # If dynamic discovery is requested, we need to collect API keys first
            api_keys_for_discovery = {}
            if use_dynamic_discovery:
                # Collect available API keys from environment
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY"
                }

                for provider_key, env_var in env_var_map.items():
                    key_value = os.getenv(env_var)
                    if key_value:
                        api_keys_for_discovery[provider_key] = key_value

            detected_provider = detect_provider_from_model(model, use_dynamic_discovery, api_keys_for_discovery)
            if not detected_provider:
                discovery_msg = " with dynamic API discovery" if use_dynamic_discovery else ""
                raise ValueError(
                    f"Could not determine provider from model '{model}'{discovery_msg}. "
                    "Please specify the provider explicitly using the 'provider' parameter."
                )
            assert detected_provider is not None  # Help MyPy understand type narrowing
            provider_name = detected_provider
            provider_enum = Provider(provider_name)

        # Determine API key
        if api_key is None:
            # Try to get from environment variables
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openrouter": "OPENROUTER_API_KEY"
            }

            env_var_name: str | None = env_var_map.get(provider_name)
            if env_var_name:
                api_key = os.getenv(env_var_name)
                if not api_key:
                    raise ValueError(
                        f"API key not provided and {env_var_name} environment variable not set. "
                        f"Please provide api_key parameter or set {env_var_name} environment variable."
                    )
            else:
                raise ValueError(
                    f"Unknown provider '{provider_name}'. Cannot determine environment variable for API key."
                )

        return cls(
            provider=provider_enum, 
            api_key=api_key, 
            request_limit=request_limit,
            token_limit=token_limit,
            max_retries=max_retries,
            base_backoff=base_backoff,
            timeout=timeout,
            **kwargs
        )

    def _init_http_clients(
        self,
        http_client: httpx.AsyncClient | None,
        sync_http_client: httpx.Client | None,
        **kwargs: Any,
    ) -> None:
        """Initialize HTTP clients with proper headers."""
        # Prepare headers
        headers = {
            "User-Agent": f"chat-limiter/0.1.0 ({self.provider.value})",
        }

        # Add provider-specific headers
        if self.api_key:
            if self.provider == Provider.OPENAI:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.provider == Provider.ANTHROPIC:
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            elif self.provider == Provider.OPENROUTER:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["HTTP-Referer"] = "https://github.com/your-repo/chat-limiter"

        # Merge with user-provided headers
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers

        # Initialize clients
        if http_client:
            self.async_client = http_client
        else:
            self.async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self._user_timeout),  # Configurable timeout
                **kwargs,
            )

        if sync_http_client:
            self.sync_client = sync_http_client
        else:
            self.sync_client = httpx.Client(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self._user_timeout),  # Configurable timeout
                **kwargs,
            )

    def _init_rate_limiters(self) -> None:
        """Initialize PyrateLimiter instances."""
        # Only initialize if we have limits
        if self.state.request_limit is None or self.state.token_limit is None:
            # Cannot initialize rate limiters without limits
            # This will be called again after limits are discovered
            self.request_limiter = None
            self.token_limiter = None
            self._effective_request_limit = None
            self._effective_token_limit = None
            return
            
        # Calculate effective limits with buffer
        effective_request_limit = int(self.state.request_limit * self.config.request_buffer_ratio)
        effective_token_limit = int(self.state.token_limit * self.config.token_buffer_ratio)
        
        # Request rate limiter
        self.request_limiter = Limiter(
            Rate(
                effective_request_limit,
                Duration.MINUTE,
            )
        )

        # Token rate limiter  
        self.token_limiter = Limiter(
            Rate(
                effective_token_limit,
                Duration.MINUTE,
            )
        )
        
        # Store effective limits for logging
        self._effective_request_limit = effective_request_limit
        self._effective_token_limit = effective_token_limit

    async def __aenter__(self) -> "ChatLimiter":
        """Async context manager entry."""
        if self._async_context_active:
            raise RuntimeError(
                "ChatLimiter is already active as an async context manager"
            )

        self._async_context_active = True

        # Discover rate limits if supported
        if self.config.supports_dynamic_limits:
            await self._discover_rate_limits()
        
        # Print rate limit information if verbose mode is enabled
        if self._verbose_mode:
            self._print_rate_limit_info()

        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Async context manager exit."""
        self._async_context_active = False
        await self.async_client.aclose()

    def __enter__(self) -> "ChatLimiter":
        """Sync context manager entry."""
        if self._sync_context_active:
            raise RuntimeError(
                "ChatLimiter is already active as a sync context manager"
            )

        self._sync_context_active = True

        # Discover rate limits if supported
        if self.config.supports_dynamic_limits:
            self._discover_rate_limits_sync()
        
        # Print rate limit information if verbose mode is enabled
        if self._verbose_mode:
            self._print_rate_limit_info()

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Sync context manager exit."""
        self._sync_context_active = False
        self.sync_client.close()

    async def _discover_rate_limits(self) -> None:
        """Discover current rate limits from the API."""
        try:
            if self.provider == Provider.OPENROUTER and self.config.auth_endpoint:
                # OpenRouter uses a special auth endpoint
                response = await self.async_client.get(self.config.auth_endpoint)
                response.raise_for_status()

                data = response.json()
                # Update limits based on response
                # This is a simplified version - actual implementation would parse the response
                logger.info(f"Discovered OpenRouter limits: {data}")

            else:
                # For other providers, we'll discover limits on first request
                if self._verbose_mode:
                    print(f"Rate limit discovery will happen on first request for {self.provider.value}")
                logger.info(
                    f"Rate limit discovery will happen on first request for {self.provider.value}"
                )

        except Exception as e:
            logger.warning(f"Failed to discover rate limits: {e}")

    def _discover_rate_limits_sync(self) -> None:
        """Sync version of rate limit discovery."""
        try:
            if self.provider == Provider.OPENROUTER and self.config.auth_endpoint:
                response = self.sync_client.get(self.config.auth_endpoint)
                response.raise_for_status()

                data = response.json()
                logger.info(f"Discovered OpenRouter limits: {data}")
            else:
                logger.info(
                    f"Rate limit discovery will happen on first request for {self.provider.value}"
                )

        except Exception as e:
            logger.warning(f"Failed to discover rate limits: {e}")

    def _update_rate_limits(self, rate_limit_info: RateLimitInfo) -> None:
        """Update rate limits based on response headers."""
        updated = False
        was_uninitialized = self.state.request_limit is None or self.state.token_limit is None

        # Update request limits
        if (
            rate_limit_info.requests_limit
            and rate_limit_info.requests_limit != self.state.request_limit
        ):
            old_limit = self.state.request_limit
            self.state.request_limit = rate_limit_info.requests_limit
            updated = True
            if was_uninitialized:
                message = f"Discovered request limit: {self.state.request_limit} req/min"
                if self._verbose_mode:
                    print(message)
                logger.info(message)
            else:
                message = f"Updated request limit: {old_limit} -> {self.state.request_limit} req/min"
                if self._verbose_mode:
                    print(message)
                logger.info(message)

        # Update token limits
        if (
            rate_limit_info.tokens_limit
            and rate_limit_info.tokens_limit != self.state.token_limit
        ):
            old_limit = self.state.token_limit
            self.state.token_limit = rate_limit_info.tokens_limit
            updated = True
            if was_uninitialized:
                message = f"Discovered token limit: {self.state.token_limit} tokens/min"
                if self._verbose_mode:
                    print(message)
                logger.info(message)
            else:
                message = f"Updated token limit: {old_limit} -> {self.state.token_limit} tokens/min"
                if self._verbose_mode:
                    print(message)
                logger.info(message)

        if updated:
            # Reinitialize rate limiters with new limits
            self._init_rate_limiters()
            
            # Update limits_discovered flag if both limits are now available
            if self.state.request_limit is not None and self.state.token_limit is not None:
                self._limits_discovered = True
            
            if was_uninitialized:
                message = "Rate limiters initialized after discovery"
                if self._verbose_mode:
                    print(message)
                    # Print updated rate limit info after discovery
                    self._print_rate_limit_info()
                logger.info(message)

        # Store the rate limit info
        self.state.last_rate_limit_info = rate_limit_info
        self.state.last_limit_update = time.time()

    def _estimate_tokens(self, request_data: dict[str, Any]) -> int:
        """Estimate token usage from request data."""
        if not self.enable_token_estimation:
            return 0

        # Simple token estimation
        # This is a placeholder - real implementation would use tiktoken or similar
        if "messages" in request_data:
            text = ""
            for message in request_data["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text += str(message["content"])

            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4

        return 0

    @asynccontextmanager
    async def _acquire_rate_limits(
        self, estimated_tokens: int = 0
    ) -> AsyncIterator[None]:
        """Acquire rate limits before making a request."""
        # Check if rate limiters are initialized
        if self.request_limiter is None or self.token_limiter is None:
            # Limits not yet discovered - this request will help discover them
            logger.info("Rate limits not yet discovered, proceeding without rate limiting for discovery")
        else:
            # Wait for request rate limit
            await asyncio.to_thread(self.request_limiter.try_acquire, "request")

            # Wait for token rate limit if we have token estimation and limiters are initialized
            if estimated_tokens > 0 and self.token_limiter is not None and self._effective_token_limit is not None:
                # Check if request is too large for bucket capacity
                if estimated_tokens > self._effective_token_limit:
                    # Log warning for large requests
                    logger.warning(
                        f"Request estimated at {estimated_tokens} tokens exceeds bucket capacity "
                        f"of {self._effective_token_limit} tokens. This may cause delays."
                    )
                    # For very large requests, we'll split the acquisition
                    # Acquire tokens in chunks to avoid bucket overflow
                    remaining_tokens = estimated_tokens
                    while remaining_tokens > 0:
                        chunk_size = min(remaining_tokens, self._effective_token_limit // 2)
                        await asyncio.to_thread(self.token_limiter.try_acquire, "token", chunk_size)
                        remaining_tokens -= chunk_size
                        if remaining_tokens > 0:
                            # Brief pause to let bucket refill
                            await asyncio.sleep(0.1)
                else:
                    # Normal acquisition for smaller requests
                    await asyncio.to_thread(self.token_limiter.try_acquire, "token", estimated_tokens)

        try:
            yield
        finally:
            # Update usage tracking
            self.state.requests_used += 1
            self.state.tokens_used += estimated_tokens
            self.state.last_request_time = time.time()

    @contextmanager
    def _acquire_rate_limits_sync(self, estimated_tokens: int = 0) -> Iterator[None]:
        """Sync version of rate limit acquisition."""
        # Check if rate limiters are initialized
        if self.request_limiter is None or self.token_limiter is None:
            # Limits not yet discovered - this request will help discover them
            logger.info("Rate limits not yet discovered, proceeding without rate limiting for discovery")
        else:
            # Wait for request rate limit
            self.request_limiter.try_acquire("request")

            # Wait for token rate limit if we have token estimation and limiters are initialized
            if estimated_tokens > 0 and self.token_limiter is not None and self._effective_token_limit is not None:
                # Check if request is too large for bucket capacity
                if estimated_tokens > self._effective_token_limit:
                    # Log warning for large requests
                    logger.warning(
                        f"Request estimated at {estimated_tokens} tokens exceeds bucket capacity "
                        f"of {self._effective_token_limit} tokens. This may cause delays."
                    )
                    # For very large requests, we'll split the acquisition
                    # Acquire tokens in chunks to avoid bucket overflow
                    remaining_tokens = estimated_tokens
                    while remaining_tokens > 0:
                        chunk_size = min(remaining_tokens, self._effective_token_limit // 2)
                        self.token_limiter.try_acquire("token", chunk_size)
                        remaining_tokens -= chunk_size
                        if remaining_tokens > 0:
                            # Brief pause to let bucket refill
                            time.sleep(0.1)
                else:
                    # Normal acquisition for smaller requests
                    self.token_limiter.try_acquire("token", estimated_tokens)

        try:
            yield
        finally:
            # Update usage tracking
            self.state.requests_used += 1
            self.state.tokens_used += estimated_tokens
            self.state.last_request_time = time.time()

    def _get_retry_decorator(self):
        """Get retry decorator with user-configured parameters."""
        return retry(
            stop=stop_after_attempt(self._user_max_retries),
            wait=wait_exponential(multiplier=self._user_base_backoff, min=1, max=60),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.ReadTimeout, httpx.ConnectTimeout)),
        )

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Wrapper that applies retry decorator dynamically."""
        try:
            return await self._get_retry_decorator()(self._request_impl)(method, url, json=json, **kwargs)
        except Exception as e:
            # Check if this is a retry error wrapping a timeout
            if hasattr(e, 'last_attempt') and e.last_attempt and e.last_attempt.exception():
                original_exception = e.last_attempt.exception()
                if isinstance(original_exception, (httpx.ReadTimeout, httpx.ConnectTimeout)):
                    # Enhance timeout error with helpful information
                    timeout_info = (
                        f"\n💡 Timeout Error Help:\n"
                        f"   Current timeout: {self._user_timeout}s\n"
                        f"   To increase timeout, use: ChatLimiter.for_model('{self.provider.value}', timeout={int(self._user_timeout + 60)})\n"
                        f"   Or reduce batch concurrency if processing multiple requests\n"
                        f"   Retries attempted: {self._user_max_retries}\n"
                    )
                    raise type(original_exception)(str(original_exception) + timeout_info) from e
            
            # For direct timeout errors (shouldn't happen due to retry decorator but just in case)
            if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)):
                timeout_info = (
                    f"\n💡 Timeout Error Help:\n"
                    f"   Current timeout: {self._user_timeout}s\n"
                    f"   To increase timeout, use: ChatLimiter.for_model('{self.provider.value}', timeout={int(self._user_timeout + 60)})\n"
                    f"   Or reduce batch concurrency if processing multiple requests\n"
                )
                raise type(e)(str(e) + timeout_info) from e
            
            # Re-raise any other exceptions unchanged
            raise

    async def _request_impl(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make an async HTTP request with rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL or path for the request
            json: JSON data to send
            **kwargs: Additional arguments passed to httpx

        Returns:
            HTTP response

        Raises:
            httpx.HTTPStatusError: For HTTP error responses
            httpx.RequestError: For request errors
        """
        if not self._async_context_active:
            raise RuntimeError("ChatLimiter must be used as an async context manager")

        # Estimate tokens if we have JSON data
        estimated_tokens = self._estimate_tokens(json or {})

        # Acquire rate limits
        async with self._acquire_rate_limits(estimated_tokens):
            # Make the request
            response = await self.async_client.request(method, url, json=json, **kwargs)

            # Extract rate limit info
            rate_limit_info = extract_rate_limit_info(
                dict(response.headers), self.config
            )

            # Update our rate limits
            if self.enable_adaptive_limits:
                self._update_rate_limits(rate_limit_info)

            # Handle rate limit errors
            if response.status_code == 429:
                self.state.consecutive_rate_limit_errors += 1
                if rate_limit_info.retry_after:
                    await asyncio.sleep(rate_limit_info.retry_after)
                else:
                    # Exponential backoff
                    backoff = self.config.base_backoff * (
                        2**self.state.consecutive_rate_limit_errors
                    )
                    await asyncio.sleep(min(backoff, self.config.max_backoff))

                response.raise_for_status()
            else:
                # Reset consecutive errors on success
                self.state.consecutive_rate_limit_errors = 0

            return response

    def request_sync(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Wrapper that applies retry decorator dynamically."""
        # For sync, we need to use the sync version of retry
        retry_decorator = retry(
            stop=stop_after_attempt(self._user_max_retries),
            wait=wait_exponential(multiplier=self._user_base_backoff, min=1, max=60),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError, httpx.ReadTimeout, httpx.ConnectTimeout)),
        )
        try:
            return retry_decorator(self._request_sync_impl)(method, url, json=json, **kwargs)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            # Enhance timeout error with helpful information
            timeout_info = (
                f"\n💡 Timeout Error Help:\n"
                f"   Current timeout: {self._user_timeout}s\n"
                f"   To increase timeout, use: ChatLimiter.for_model('{self.provider.value}', timeout={int(self._user_timeout + 60)})\n"
                f"   Or reduce batch concurrency if processing multiple requests\n"
            )
            raise type(e)(str(e) + timeout_info) from e

    def _request_sync_impl(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make a sync HTTP request with rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL or path for the request
            json: JSON data to send
            **kwargs: Additional arguments passed to httpx

        Returns:
            HTTP response

        Raises:
            httpx.HTTPStatusError: For HTTP error responses
            httpx.RequestError: For request errors
        """
        if not self._sync_context_active:
            raise RuntimeError("ChatLimiter must be used as a sync context manager")

        # Estimate tokens if we have JSON data
        estimated_tokens = self._estimate_tokens(json or {})

        # Acquire rate limits
        with self._acquire_rate_limits_sync(estimated_tokens):
            # Make the request
            response = self.sync_client.request(method, url, json=json, **kwargs)

            # Extract rate limit info
            rate_limit_info = extract_rate_limit_info(
                dict(response.headers), self.config
            )

            # Update our rate limits
            if self.enable_adaptive_limits:
                self._update_rate_limits(rate_limit_info)

            # Handle rate limit errors
            if response.status_code == 429:
                self.state.consecutive_rate_limit_errors += 1
                if rate_limit_info.retry_after:
                    time.sleep(rate_limit_info.retry_after)
                else:
                    # Exponential backoff
                    backoff = self.config.base_backoff * (
                        2**self.state.consecutive_rate_limit_errors
                    )
                    time.sleep(min(backoff, self.config.max_backoff))

                response.raise_for_status()
            else:
                # Reset consecutive errors on success
                self.state.consecutive_rate_limit_errors = 0

            return response

    def get_current_limits(self) -> dict[str, Any]:
        """Get current rate limit information."""
        return {
            "provider": self.provider.value,
            "request_limit": self.state.request_limit,
            "token_limit": self.state.token_limit,
            "requests_used": self.state.requests_used,
            "tokens_used": self.state.tokens_used,
            "last_request_time": self.state.last_request_time,
            "last_limit_update": self.state.last_limit_update,
            "consecutive_rate_limit_errors": self.state.consecutive_rate_limit_errors,
        }

    def reset_usage_tracking(self) -> None:
        """Reset usage tracking counters."""
        self.state.requests_used = 0
        self.state.tokens_used = 0
        self.state.consecutive_rate_limit_errors = 0

    # High-level chat completion methods

    async def chat_completion(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """
        Make a high-level chat completion request.

        Args:
            model: The model to use for completion
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatCompletionResponse with the completion result

        Raises:
            ValueError: If provider cannot be determined from model
            httpx.HTTPStatusError: For HTTP error responses
            httpx.RequestError: For request errors
        """
        if not self._async_context_active:
            raise RuntimeError("ChatLimiter must be used as an async context manager")

        # Create request object
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
            **kwargs
        )

        # Get the appropriate adapter
        adapter = get_adapter(self.provider)

        # Format the request for the provider
        formatted_request = adapter.format_request(request)

        # Make the HTTP request
        response = await self.request(
            "POST",
            adapter.get_endpoint(),
            json=formatted_request
        )

        # Parse the response
        response_data = response.json()
        return adapter.parse_response(response_data, request)

    def chat_completion_sync(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """
        Make a synchronous high-level chat completion request.

        Args:
            model: The model to use for completion
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatCompletionResponse with the completion result

        Raises:
            ValueError: If provider cannot be determined from model
            httpx.HTTPStatusError: For HTTP error responses
            httpx.RequestError: For request errors
        """
        if not self._sync_context_active:
            raise RuntimeError("ChatLimiter must be used as a sync context manager")

        # Create request object
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
            **kwargs
        )

        # Get the appropriate adapter
        adapter = get_adapter(self.provider)

        # Format the request for the provider
        formatted_request = adapter.format_request(request)

        # Make the HTTP request
        response = self.request_sync(
            "POST",
            adapter.get_endpoint(),
            json=formatted_request
        )

        # Parse the response
        response_data = response.json()
        return adapter.parse_response(response_data, request)

    # Convenience methods for different message types

    async def simple_chat(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple chat completion that returns just the text response.

        Args:
            model: The model to use
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            The text response from the model
        """
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await self.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if response.choices:
            return response.choices[0].message.content
        return ""

    def simple_chat_sync(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple synchronous chat completion that returns just the text response.

        Args:
            model: The model to use
            prompt: The user prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            The text response from the model
        """
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = self.chat_completion_sync(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if response.choices:
            return response.choices[0].message.content
        return ""
    
    def set_verbose_mode(self, verbose: bool) -> None:
        """Set verbose mode for detailed logging."""
        self._verbose_mode = verbose
    
    def _print_rate_limit_info(self) -> None:
        """Print current rate limit configuration."""
        print(f"\n=== Rate Limit Configuration for {self.provider.value.title()} ===")
        print(f"Provider: {self.provider.value}")
        print(f"Base URL: {self.config.base_url}")
        
        # Handle None values for limits
        if self.state.request_limit is not None:
            effective_req = self._effective_request_limit or "not calculated"
            print(f"Request Limit: {self.state.request_limit}/minute (effective: {effective_req}/minute)")
        else:
            print("Request Limit: Not yet discovered (will be fetched from API)")
            
        if self.state.token_limit is not None:
            effective_tok = self._effective_token_limit or "not calculated"
            print(f"Token Limit: {self.state.token_limit}/minute (effective: {effective_tok}/minute)")
        else:
            print("Token Limit: Not yet discovered (will be fetched from API)")
            
        print(f"Request Buffer Ratio: {self.config.request_buffer_ratio}")
        print(f"Token Buffer Ratio: {self.config.token_buffer_ratio}")
        print(f"Adaptive Limits: {self.enable_adaptive_limits}")
        print(f"Token Estimation: {self.enable_token_estimation}")
        print(f"Dynamic Discovery: {self.config.supports_dynamic_limits}")
        print(f"Limits Discovered: {self._limits_discovered}")
        print("=" * 50)
