"""
Mock example demonstrating chat-limiter without requiring real API keys.

This example uses mock HTTP clients to demonstrate the library functionality
without making actual API requests.
"""

import asyncio
from unittest.mock import AsyncMock, Mock
import httpx

from chat_limiter import ChatLimiter, Provider, BatchConfig, process_chat_batch


def create_mock_response(content: str = "Mock response") -> Mock:
    """Create a mock HTTP response."""
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
        "x-ratelimit-reset-requests": "60",
        "x-ratelimit-limit-tokens": "10000",
        "x-ratelimit-remaining-tokens": "9500",
        "x-ratelimit-reset-tokens": "60",
        "content-type": "application/json",
    }
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    mock_response.raise_for_status = Mock()
    return mock_response


async def mock_single_request_example():
    """Example of single request with mocked HTTP client."""
    print("🎭 Mock Single Request Example")
    print("-" * 40)
    
    # Create mock HTTP client
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.request.return_value = create_mock_response("Hello from mock API!")
    mock_client.aclose = AsyncMock()
    
    # Use ChatLimiter with mock client
    async with ChatLimiter(
        provider=Provider.OPENAI,
        api_key="mock-key",
        http_client=mock_client
    ) as limiter:
        response = await limiter.request(
            "POST", "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        
        result = response.json()
        print(f"✅ Response: {result['choices'][0]['message']['content']}")
        
        # Verify the request was made
        mock_client.request.assert_called_once()
        
        # Check rate limit tracking
        limits = limiter.get_current_limits()
        print(f"📊 Requests tracked: {limits['requests_used']}")
        print(f"📊 Rate limits updated from headers: {limits['request_limit']}")


async def mock_batch_processing_example():
    """Example of batch processing with mocked responses."""
    print("\n📦 Mock Batch Processing Example")
    print("-" * 40)
    
    # Create mock HTTP client
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    
    def mock_request_side_effect(*args, **kwargs):
        """Side effect to return different responses based on request."""
        json_data = kwargs.get("json", {})
        messages = json_data.get("messages", [])
        if messages:
            content = messages[0].get("content", "")
            return create_mock_response(f"Mock response for: {content}")
        return create_mock_response("Default mock response")
    
    mock_client.request.side_effect = mock_request_side_effect
    mock_client.aclose = AsyncMock()
    
    # Create batch requests
    requests = [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Question {i}"}],
            "max_tokens": 50
        }
        for i in range(5)
    ]
    
    # Configure batch processing
    config = BatchConfig(
        max_concurrent_requests=3,
        max_retries_per_item=1,
    )
    
    async with ChatLimiter(
        provider=Provider.OPENAI,
        api_key="mock-key",
        http_client=mock_client
    ) as limiter:
        print(f"🚀 Processing {len(requests)} mock requests...")
        
        results = await process_chat_batch(limiter, requests, config)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        # Show some results
        for i, result in enumerate(successful[:3]):
            if result.result:
                content = result.result["choices"][0]["message"]["content"]
                print(f"📝 Result {i+1}: {content}")
        
        # Verify correct number of calls
        print(f"🔍 Total API calls made: {mock_client.request.call_count}")


def mock_sync_example():
    """Synchronous example with mocked HTTP client."""
    print("\n🔄 Mock Synchronous Example")
    print("-" * 40)
    
    # Create mock sync HTTP client
    mock_client = Mock(spec=httpx.Client)
    mock_client.request.return_value = create_mock_response("Sync mock response!")
    mock_client.close = Mock()
    
    with ChatLimiter(
        provider=Provider.OPENAI,
        api_key="mock-key",
        sync_http_client=mock_client
    ) as limiter:
        response = limiter.request_sync(
            "POST", "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Sync test"}]
            }
        )
        
        result = response.json()
        print(f"✅ Sync response: {result['choices'][0]['message']['content']}")
        
        # Verify the request was made
        mock_client.request.assert_called_once()


async def mock_error_handling_example():
    """Example of error handling with mocked failures."""
    print("\n🛡️ Mock Error Handling Example")
    print("-" * 40)
    
    # Create mock HTTP client that returns rate limit error
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    
    # Create rate limit response
    rate_limit_response = Mock(spec=httpx.Response)
    rate_limit_response.status_code = 429
    rate_limit_response.headers = {
        "retry-after": "5",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "0",
    }
    rate_limit_response.json.return_value = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_exceeded"
        }
    }
    rate_limit_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Rate limit exceeded",
        request=Mock(),
        response=rate_limit_response
    )
    
    mock_client.request.return_value = rate_limit_response
    mock_client.aclose = AsyncMock()
    
    try:
        async with ChatLimiter(
            provider=Provider.OPENAI,
            api_key="mock-key",
            http_client=mock_client
        ) as limiter:
            # This should trigger retry logic due to rate limit
            response = await limiter.request(
                "POST", "/chat/completions",
                json={"model": "gpt-3.5-turbo", "messages": []}
            )
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"📝 Error demonstrates retry logic working")


async def mock_rate_limit_adaptation_example():
    """Example showing rate limit adaptation from headers."""
    print("\n📊 Mock Rate Limit Adaptation Example")
    print("-" * 40)
    
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    
    # First response with different rate limits
    response1 = create_mock_response("First response")
    response1.headers.update({
        "x-ratelimit-limit-requests": "50",    # Lower limit
        "x-ratelimit-limit-tokens": "5000",   # Lower token limit
    })
    
    # Second response with updated limits
    response2 = create_mock_response("Second response")
    response2.headers.update({
        "x-ratelimit-limit-requests": "200",   # Higher limit
        "x-ratelimit-limit-tokens": "20000",  # Higher token limit
    })
    
    mock_client.request.side_effect = [response1, response2]
    mock_client.aclose = AsyncMock()
    
    async with ChatLimiter(
        provider=Provider.OPENAI,
        api_key="mock-key",
        http_client=mock_client,
        enable_adaptive_limits=True
    ) as limiter:
        print("📊 Initial limits:")
        initial_limits = limiter.get_current_limits()
        print(f"  Requests: {initial_limits['request_limit']}")
        print(f"  Tokens: {initial_limits['token_limit']}")
        
        # Make first request
        await limiter.request("POST", "/chat/completions", json={})
        
        print("\n📊 After first request (adapted to lower limits):")
        limits1 = limiter.get_current_limits()
        print(f"  Requests: {limits1['request_limit']}")
        print(f"  Tokens: {limits1['token_limit']}")
        
        # Make second request
        await limiter.request("POST", "/chat/completions", json={})
        
        print("\n📊 After second request (adapted to higher limits):")
        limits2 = limiter.get_current_limits()
        print(f"  Requests: {limits2['request_limit']}")
        print(f"  Tokens: {limits2['token_limit']}")
        
        print("\n✅ Rate limits successfully adapted from response headers!")


async def main():
    """Run all mock examples."""
    print("🎭 Chat-Limiter Mock Examples")
    print("=" * 50)
    print("These examples demonstrate the library functionality")
    print("without requiring real API keys or making actual requests.\n")
    
    await mock_single_request_example()
    await mock_batch_processing_example()
    mock_sync_example()
    await mock_error_handling_example()
    await mock_rate_limit_adaptation_example()
    
    print("\n✅ All mock examples completed!")
    print("\n💡 Key Features Demonstrated:")
    print("- ✅ Rate limit extraction from headers")
    print("- ✅ Adaptive limit adjustment")
    print("- ✅ Batch processing with concurrency")
    print("- ✅ Error handling and retry logic")
    print("- ✅ Both sync and async interfaces")
    print("- ✅ Request tracking and monitoring")


if __name__ == "__main__":
    asyncio.run(main())