# chat-limiter

A Pythonic rate limiter for OpenAI, Anthropic, and OpenRouter APIs that automatically fetches and respects provider limits while maximizing throughput.

## Features

- =€ **Automatic Rate Limit Discovery**: Fetches current limits from API response headers
- = **Sync & Async Support**: Use with `async/await` or synchronous code
- =æ **Batch Processing**: Process multiple requests efficiently with concurrency control
- =á **Intelligent Retry Logic**: Exponential backoff with provider-specific optimizations
- <¯ **Multi-Provider Support**: Works seamlessly with OpenAI, Anthropic, and OpenRouter
- <× **Pythonic Design**: Context manager interface with proper error handling
- >ê **Fully Tested**: Comprehensive test suite with 91% coverage
- =Ê **Token Estimation**: Basic token counting for better rate limit management

## Installation

```bash
pip install chat-limiter
```

Or with uv:

```bash
uv add chat-limiter
```

## Quick Start

### Basic Usage

```python
import asyncio
from chat_limiter import ChatLimiter, Provider

async def main():
    # Initialize the limiter
    async with ChatLimiter(
        provider=Provider.OPENAI,
        api_key="sk-your-openai-key"
    ) as limiter:
        # Make rate-limited requests
        response = await limiter.request(
            "POST", "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        
        result = response.json()
        print(result["choices"][0]["message"]["content"])

asyncio.run(main())
```

### Synchronous Usage

```python
from chat_limiter import ChatLimiter, Provider

with ChatLimiter(
    provider=Provider.ANTHROPIC,
    api_key="sk-ant-your-key"
) as limiter:
    response = limiter.request_sync(
        "POST", "/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello Claude!"}]
        }
    )
    
    result = response.json()
    print(result["content"][0]["text"])
```

### Batch Processing

```python
import asyncio
from chat_limiter import ChatLimiter, Provider, process_chat_batch, BatchConfig

async def batch_example():
    requests = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"Question {i}"}]}
        for i in range(10)
    ]
    
    async with ChatLimiter(provider=Provider.OPENAI, api_key="sk-key") as limiter:
        # Process with custom configuration
        config = BatchConfig(
            max_concurrent_requests=5,
            max_retries_per_item=3,
            group_by_model=True
        )
        
        results = await process_chat_batch(limiter, requests, config)
        
        # Check results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"Successful: {len(successful)}, Failed: {len(failed)}")

asyncio.run(batch_example())
```

## Provider Support

### OpenAI

-  Automatic header parsing (`x-ratelimit-*`)
-  Request and token rate limiting
-  Exponential backoff with jitter
-  Model-specific optimizations

### Anthropic

-  Claude-specific headers (`anthropic-ratelimit-*`)
-  Separate input/output token tracking
-  Tier-based rate limit handling
-  Retry-after header support

### OpenRouter

-  Multi-model proxy support
-  Dynamic limit discovery via auth endpoint
-  Model-specific rate adjustments
-  Credit-based limiting

## Configuration

### Provider Configuration

```python
from chat_limiter import ChatLimiter, ProviderConfig, Provider

# Custom provider configuration
config = ProviderConfig(
    provider=Provider.OPENAI,
    base_url="https://api.openai.com/v1",
    default_request_limit=100,
    default_token_limit=50000,
    max_retries=5,
    base_backoff=2.0,
    request_buffer_ratio=0.8  # Use 80% of limits
)

async with ChatLimiter(config=config, api_key="sk-key") as limiter:
    # Your requests here
    pass
```

### Batch Configuration

```python
from chat_limiter import BatchConfig

config = BatchConfig(
    max_concurrent_requests=10,     # Concurrent request limit
    max_workers=4,                  # Thread pool size for sync
    max_retries_per_item=3,         # Retries per failed item
    retry_delay=1.0,                # Base retry delay
    stop_on_first_error=False,      # Continue on individual failures
    group_by_model=True,            # Group requests by model
    adaptive_batch_size=True        # Adapt batch size to rate limits
)
```

## Advanced Usage

### Custom HTTP Clients

```python
import httpx
from chat_limiter import ChatLimiter, Provider

# Use custom HTTP client
custom_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    headers={"Custom-Header": "value"}
)

async with ChatLimiter(
    provider=Provider.OPENAI,
    api_key="sk-key",
    http_client=custom_client
) as limiter:
    # Requests will use your custom client
    response = await limiter.request("GET", "/models")
```

### Error Handling

```python
from chat_limiter import ChatLimiter, Provider
from tenacity import RetryError
import httpx

async with ChatLimiter(provider=Provider.OPENAI, api_key="sk-key") as limiter:
    try:
        response = await limiter.request("POST", "/chat/completions", json=data)
    except RetryError as e:
        print(f"Request failed after retries: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        print(f"Request error: {e}")
```

### Monitoring and Metrics

```python
from chat_limiter import ChatLimiter, Provider

async with ChatLimiter(provider=Provider.OPENAI, api_key="sk-key") as limiter:
    # Make some requests...
    await limiter.request("POST", "/chat/completions", json=data)
    
    # Check current limits and usage
    limits = limiter.get_current_limits()
    print(f"Requests used: {limits['requests_used']}/{limits['request_limit']}")
    print(f"Tokens used: {limits['tokens_used']}/{limits['token_limit']}")
    
    # Reset usage tracking
    limiter.reset_usage_tracking()
```

### Batch Processing with Custom Logic

```python
from chat_limiter import BatchProcessor, ChatLimiter, BatchItem, BatchResult
from typing import Dict, Any

class CustomBatchProcessor(BatchProcessor[Dict[str, Any], Dict[str, Any]]):
    async def process_item(self, item: BatchItem[Dict[str, Any]]) -> Dict[str, Any]:
        # Custom processing logic
        response = await self.limiter.request(
            "POST", "/custom/endpoint", 
            json=item.data
        )
        return response.json()
    
    def process_item_sync(self, item: BatchItem[Dict[str, Any]]) -> Dict[str, Any]:
        # Sync version
        response = self.limiter.request_sync(
            "POST", "/custom/endpoint",
            json=item.data
        )
        return response.json()

# Use custom processor
async with ChatLimiter(provider=Provider.OPENAI, api_key="sk-key") as limiter:
    processor = CustomBatchProcessor(limiter)
    results = await processor.process_batch(your_data)
```

## Rate Limiting Details

### How It Works

1. **Header Parsing**: Automatically extracts rate limit information from API response headers
2. **Token Bucket Algorithm**: Uses PyrateLimiter for smooth rate limiting with burst support
3. **Adaptive Limits**: Updates limits based on server responses in real-time
4. **Intelligent Queuing**: Coordinates requests to stay under limits while maximizing throughput

### Provider-Specific Behavior

| Provider   | Request Limits | Token Limits | Dynamic Discovery | Special Features |
|------------|---------------|--------------|-------------------|------------------|
| OpenAI     |  RPM        |  TPM       |  Headers        | Model detection, batch optimization |
| Anthropic  |  RPM        |  Input/Output TPM |  Headers | Tier handling, thinking models |
| OpenRouter |  RPM        |  TPM       |  Auth endpoint  | Multi-model, credit tracking |

## Testing

The library includes a comprehensive test suite:

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=chat_limiter

# Run specific test file
uv run pytest tests/test_limiter.py -v
```

## Development

```bash
# Clone the repository
git clone https://github.com/your-repo/chat-limiter.git
cd chat-limiter

# Install dependencies
uv sync --group dev

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run mypy src/

# Format code
uv run black src/ tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### 0.1.0 (Initial Release)

- Multi-provider support (OpenAI, Anthropic, OpenRouter)
- Async and sync interfaces
- Batch processing with concurrency control
- Automatic rate limit discovery
- Comprehensive test suite
- Type hints and documentation