# chat-limiter

A Pythonic rate limiter for OpenAI, Anthropic, and OpenRouter APIs that provides a high-level chat completion interface with automatic rate limit management.

## Features

- 🚀 **High-Level Chat Interface**: OpenAI/Anthropic-style chat completion methods
- 📡 **Automatic Rate Limit Discovery**: Fetches current limits from API response headers
- ⚡ **Sync & Async Support**: Use with `async/await` or synchronous code
- 📦 **Batch Processing**: Process multiple requests efficiently with concurrency control
- 🔄 **Intelligent Retry Logic**: Exponential backoff with provider-specific optimizations
- 🌐 **Multi-Provider Support**: Works seamlessly with OpenAI, Anthropic, and OpenRouter
- 🎯 **Pythonic Design**: Context manager interface with proper error handling
- 🛡️ **Fully Tested**: Comprehensive test suite with 93% coverage
- 🔧 **Token Estimation**: Basic token counting for better rate limit management
- 🔑 **Environment Variable Support**: Automatic API key detection from env vars
- 🔀 **Provider Override**: Manually specify provider for custom models

## Installation

```bash
pip install chat-limiter
```

Or with uv:

```bash
uv add chat-limiter
```

## Quick Start

### High-Level Chat Completion Interface (Recommended)

```python
import asyncio
from chat_limiter import ChatLimiter, Message, MessageRole

async def main():
    # Auto-detect provider and use environment variable for API key
    async with ChatLimiter.for_model("gpt-4o") as limiter:
        response = await limiter.chat_completion(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )
        print(response.choices[0].message.content)

    # Or provide API key explicitly
    async with ChatLimiter.for_model("claude-3-5-sonnet-20241022", api_key="sk-ant-...") as limiter:
        response = await limiter.simple_chat(
            model="claude-3-5-sonnet-20241022",
            prompt="What is Python?",
            max_tokens=100
        )
        print(response)

asyncio.run(main())
```

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"  
export OPENROUTER_API_KEY="sk-or-your-openrouter-key"
```

The library will automatically detect the provider from the model name and use the appropriate environment variable.

### Provider Override

For custom models or when auto-detection fails:

```python
async with ChatLimiter.for_model(
    "custom-model-name",
    provider="openai",  # or "anthropic", "openrouter"
    api_key="sk-key"
) as limiter:
    response = await limiter.chat_completion(
        model="custom-model-name",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
```

### Synchronous Usage

```python
from chat_limiter import ChatLimiter, Message, MessageRole

with ChatLimiter.for_model("gpt-4o") as limiter:
    response = limiter.chat_completion_sync(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
    print(response.choices[0].message.content)

    # Or use the simple interface
    text_response = limiter.simple_chat_sync(
        model="gpt-4o",
        prompt="What is the capital of France?",
        max_tokens=50
    )
    print(text_response)
```

### Batch Processing with High-Level Interface

```python
import asyncio
from chat_limiter import (
    ChatLimiter, 
    Message, 
    MessageRole, 
    ChatCompletionRequest,
    process_chat_completion_batch,
    create_chat_completion_requests,
    BatchConfig
)

async def batch_example():
    # Create requests from simple prompts
    requests = create_chat_completion_requests(
        model="gpt-4o",
        prompts=["Hello!", "How are you?", "What is Python?"],
        max_tokens=50,
        temperature=0.7
    )
    
    async with ChatLimiter.for_model("gpt-4o") as limiter:
        # Process with custom configuration
        config = BatchConfig(
            max_concurrent_requests=5,
            max_retries_per_item=3,
            group_by_model=True
        )
        
        results = await process_chat_completion_batch(limiter, requests, config)
        
        # Extract successful responses
        for result in results:
            if result.success:
                response = result.result
                print(response.choices[0].message.content)

asyncio.run(batch_example())
```

## Provider Support

### Auto-Detection from Model Names

The library automatically detects providers based on model names:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, etc.
- **OpenRouter**: `openai/gpt-4o`, `anthropic/claude-3-sonnet`, etc.

### Provider-Specific Features

**OpenAI**
- ✅ Automatic header parsing (`x-ratelimit-*`)
- ✅ Request and token rate limiting  
- ✅ Exponential backoff with jitter
- ✅ Model-specific optimizations

**Anthropic**
- ✅ Claude-specific headers (`anthropic-ratelimit-*`)
- ✅ Separate input/output token tracking
- ✅ System message handling
- ✅ Retry-after header support

**OpenRouter**
- ✅ Multi-model proxy support
- ✅ Dynamic limit discovery
- ✅ Model-specific rate adjustments
- ✅ Credit-based limiting

## Advanced Usage

### Low-Level Interface

For advanced users who need direct HTTP access:

```python
from chat_limiter import ChatLimiter, Provider

async with ChatLimiter(
    provider=Provider.OPENAI,
    api_key="sk-your-key"
) as limiter:
    # Direct HTTP requests
    response = await limiter.request(
        "POST", "/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
    )
    
    result = response.json()
    print(result["choices"][0]["message"]["content"])
```

### Custom HTTP Clients

```python
import httpx
from chat_limiter import ChatLimiter

# Use custom HTTP client
custom_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    headers={"Custom-Header": "value"}
)

async with ChatLimiter.for_model(
    "gpt-4o",
    http_client=custom_client
) as limiter:
    response = await limiter.chat_completion(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
```

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
    response = await limiter.chat_completion(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
```

### Error Handling

```python
from chat_limiter import ChatLimiter, Message, MessageRole
from tenacity import RetryError
import httpx

async with ChatLimiter.for_model("gpt-4o") as limiter:
    try:
        response = await limiter.chat_completion(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Hello!")]
        )
    except RetryError as e:
        print(f"Request failed after retries: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError as e:
        print(f"Request error: {e}")
```

### Monitoring and Metrics

```python
async with ChatLimiter.for_model("gpt-4o") as limiter:
    # Make some requests...
    await limiter.chat_completion(
        model="gpt-4o",
        messages=[Message(role=MessageRole.USER, content="Hello!")]
    )
    
    # Check current limits and usage
    limits = limiter.get_current_limits()
    print(f"Requests used: {limits['requests_used']}/{limits['request_limit']}")
    print(f"Tokens used: {limits['tokens_used']}/{limits['token_limit']}")
    
    # Reset usage tracking
    limiter.reset_usage_tracking()
```

## Message Types and Parameters

### Message Structure

```python
from chat_limiter import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Hello!"),
    Message(role=MessageRole.ASSISTANT, content="Hi there!"),
    Message(role=MessageRole.USER, content="How are you?")
]
```

### Chat Completion Parameters

```python
response = await limiter.chat_completion(
    model="gpt-4o",
    messages=messages,
    max_tokens=100,           # Maximum tokens to generate
    temperature=0.7,          # Sampling temperature (0-2)
    top_p=0.9,               # Top-p sampling
    stop=["END"],            # Stop sequences
    stream=False,            # Streaming response
    frequency_penalty=0.0,   # Frequency penalty (-2 to 2)
    presence_penalty=0.0,    # Presence penalty (-2 to 2)
    top_k=40,               # Top-k sampling (Anthropic/OpenRouter)
)
```

## Batch Processing

### Simple Batch Processing

```python
from chat_limiter import create_chat_completion_requests, process_chat_completion_batch

# Create requests from prompts
requests = create_chat_completion_requests(
    model="gpt-4o",
    prompts=["Question 1", "Question 2", "Question 3"],
    max_tokens=50
)

async with ChatLimiter.for_model("gpt-4o") as limiter:
    results = await process_chat_completion_batch(limiter, requests)
    
    # Process results
    for result in results:
        if result.success:
            print(result.result.choices[0].message.content)
        else:
            print(f"Error: {result.error}")
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

## Rate Limiting Details

### How It Works

1. **Header Parsing**: Automatically extracts rate limit information from API response headers
2. **Token Bucket Algorithm**: Uses PyrateLimiter for smooth rate limiting with burst support
3. **Adaptive Limits**: Updates limits based on server responses in real-time
4. **Intelligent Queuing**: Coordinates requests to stay under limits while maximizing throughput

### Provider-Specific Behavior

| Provider   | Request Limits | Token Limits | Dynamic Discovery | Special Features |
|------------|---------------|--------------|-------------------|------------------|
| OpenAI     | ✅ RPM        | ✅ TPM       | ✅ Headers        | Model detection, batch optimization |
| Anthropic  | ✅ RPM        | ✅ Input/Output TPM | ✅ Headers | Tier handling, system messages |
| OpenRouter | ✅ RPM        | ✅ TPM       | ✅ Auth endpoint  | Multi-model, credit tracking |

## Testing

The library includes a comprehensive test suite:

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=chat_limiter

# Run specific test file
uv run pytest tests/test_high_level_interface.py -v
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
uv run ruff format src/ tests/
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

### 0.2.0 (Latest)

- 🚀 **High-level chat completion interface** - OpenAI/Anthropic-style methods
- 🔑 **Environment variable support** - Automatic API key detection
- 🔀 **Provider override** - Manual provider specification for custom models
- 📦 **Enhanced batch processing** - High-level batch operations with ChatCompletionRequest
- 🎯 **Unified message types** - Cross-provider message and response compatibility
- 🧪 **Improved testing** - 93% test coverage with comprehensive high-level interface tests

### 0.1.0 (Initial Release)

- Multi-provider support (OpenAI, Anthropic, OpenRouter)
- Async and sync interfaces
- Batch processing with concurrency control
- Automatic rate limit discovery
- Comprehensive test suite
- Type hints and documentation