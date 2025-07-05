"""
Basic usage examples for chat-limiter.

This example demonstrates the core functionality of the chat-limiter library
including single requests, batch processing, and error handling.
"""

import asyncio
import os
from typing import List, Dict, Any

from chat_limiter import (
    ChatLimiter,
    Provider,
    BatchConfig,
    process_chat_batch,
    process_chat_batch_sync,
)


async def basic_openai_example():
    """Basic example using OpenAI API."""
    print("🤖 Basic OpenAI Example")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    async with ChatLimiter(provider=Provider.OPENAI, api_key=api_key) as limiter:
        response = await limiter.request(
            "POST", "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 50
            }
        )
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        print(f"✅ Response: {answer}")
        
        # Check rate limit status
        limits = limiter.get_current_limits()
        print(f"📊 Requests used: {limits['requests_used']}")
        print(f"📊 Tokens used: {limits['tokens_used']}")


def sync_anthropic_example():
    """Synchronous example using Anthropic API."""
    print("\n🧠 Synchronous Anthropic Example")
    print("-" * 40)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return
    
    with ChatLimiter(provider=Provider.ANTHROPIC, api_key=api_key) as limiter:
        response = limiter.request_sync(
            "POST", "/messages",
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "What is Python?"}
                ]
            }
        )
        
        result = response.json()
        answer = result["content"][0]["text"]
        print(f"✅ Response: {answer}")
        
        # Check rate limit status
        limits = limiter.get_current_limits()
        print(f"📊 Requests used: {limits['requests_used']}")


async def batch_processing_example():
    """Example of batch processing multiple requests."""
    print("\n📦 Batch Processing Example")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    # Create a batch of requests
    questions = [
        "What is machine learning?",
        "Explain quantum computing",
        "What are neural networks?",
        "How does blockchain work?",
        "What is artificial intelligence?"
    ]
    
    requests = [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 100
        }
        for question in questions
    ]
    
    # Configure batch processing
    config = BatchConfig(
        max_concurrent_requests=3,  # Process 3 requests concurrently
        max_retries_per_item=2,     # Retry failed requests up to 2 times
        group_by_model=True,        # Group requests by model
    )
    
    async with ChatLimiter(provider=Provider.OPENAI, api_key=api_key) as limiter:
        print(f"🚀 Processing {len(requests)} requests...")
        
        results = await process_chat_batch(limiter, requests, config)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        # Show first successful result
        if successful:
            first_result = successful[0].result
            if first_result:
                answer = first_result["choices"][0]["message"]["content"]
                print(f"📝 First answer: {answer[:100]}...")
        
        # Show processing statistics
        if hasattr(limiter, '_batch_processor'):
            stats = results[0].item.metadata if results else {}
            print(f"📊 Processing stats available in result metadata")


async def error_handling_example():
    """Example of proper error handling."""
    print("\n🛡️ Error Handling Example")
    print("-" * 40)
    
    # Use an invalid API key to demonstrate error handling
    invalid_key = "sk-invalid-key-for-demo"
    
    try:
        async with ChatLimiter(provider=Provider.OPENAI, api_key=invalid_key) as limiter:
            response = await limiter.request(
                "POST", "/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test"}]
                }
            )
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"📝 Error message: {str(e)[:100]}...")


async def custom_configuration_example():
    """Example of custom configuration."""
    print("\n⚙️ Custom Configuration Example")
    print("-" * 40)
    
    from chat_limiter import ProviderConfig
    
    # Create custom configuration with conservative limits
    custom_config = ProviderConfig(
        provider=Provider.OPENAI,
        base_url="https://api.openai.com/v1",
        default_request_limit=30,   # Conservative request limit
        default_token_limit=15000,  # Conservative token limit
        max_retries=5,              # More retries
        base_backoff=2.0,           # Longer backoff
        request_buffer_ratio=0.8,   # Use only 80% of limits
        token_buffer_ratio=0.8,
    )
    
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    
    print(f"🔧 Using custom config:")
    print(f"   - Request limit: {custom_config.default_request_limit}")
    print(f"   - Token limit: {custom_config.default_token_limit}")
    print(f"   - Buffer ratio: {custom_config.request_buffer_ratio}")
    print(f"   - Max retries: {custom_config.max_retries}")
    
    # Note: This won't make actual requests with demo key
    async with ChatLimiter(config=custom_config, api_key=api_key) as limiter:
        limits = limiter.get_current_limits()
        print(f"✅ Limiter initialized with custom config")
        print(f"📊 Provider: {limits['provider']}")


def sync_batch_example():
    """Synchronous batch processing example."""
    print("\n🔄 Synchronous Batch Processing")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    requests = [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Count to {i}"}],
            "max_tokens": 50
        }
        for i in range(1, 4)  # Smaller batch for demo
    ]
    
    with ChatLimiter(provider=Provider.OPENAI, api_key=api_key) as limiter:
        results = process_chat_batch_sync(limiter, requests)
        
        successful = [r for r in results if r.success]
        print(f"✅ Processed {len(successful)} requests synchronously")


async def monitoring_example():
    """Example of monitoring rate limits and usage."""
    print("\n📊 Monitoring Example")
    print("-" * 40)
    
    api_key = os.getenv("OPENAI_API_KEY", "demo-key")
    
    async with ChatLimiter(provider=Provider.OPENAI, api_key=api_key) as limiter:
        print("Initial state:")
        limits = limiter.get_current_limits()
        for key, value in limits.items():
            print(f"  {key}: {value}")
        
        # Simulate some usage
        limiter.state.requests_used = 10
        limiter.state.tokens_used = 5000
        
        print("\nAfter simulated usage:")
        limits = limiter.get_current_limits()
        print(f"  Requests: {limits['requests_used']}/{limits['request_limit']}")
        print(f"  Tokens: {limits['tokens_used']}/{limits['token_limit']}")
        
        # Reset tracking
        limiter.reset_usage_tracking()
        
        print("\nAfter reset:")
        limits = limiter.get_current_limits()
        print(f"  Requests: {limits['requests_used']}")
        print(f"  Tokens: {limits['tokens_used']}")


async def main():
    """Run all examples."""
    print("🚀 Chat-Limiter Examples")
    print("=" * 50)
    
    # Basic examples
    await basic_openai_example()
    sync_anthropic_example()
    
    # Batch processing
    await batch_processing_example()
    sync_batch_example()
    
    # Advanced examples
    await error_handling_example()
    await custom_configuration_example()
    await monitoring_example()
    
    print("\n✅ All examples completed!")
    print("\n💡 Tips:")
    print("- Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables")
    print("- Adjust rate limits based on your API tier")
    print("- Use batch processing for multiple requests")
    print("- Monitor usage with get_current_limits()")


if __name__ == "__main__":
    asyncio.run(main())