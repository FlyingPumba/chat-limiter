"""
High-level batch processing examples for chat-limiter.

This example demonstrates the new high-level batch processing interface that
works with ChatCompletionRequest and ChatCompletionResponse objects.
"""

import asyncio
import os
from typing import List

from chat_limiter import (
    ChatLimiter,
    Message,
    MessageRole,
    ChatCompletionRequest,
    BatchConfig,
    process_chat_completion_batch,
    process_chat_completion_batch_sync,
    create_chat_completion_requests,
)


async def simple_batch_example():
    """Simple batch processing example using high-level interface."""
    print("📦 Simple High-Level Batch Example")
    print("-" * 45)
    
    # Create requests using the helper function
    requests = create_chat_completion_requests(
        model="gpt-4o",
        prompts=[
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?",
            "What is the capital of Spain?",
        ],
        max_tokens=20,
        temperature=0.7
    )
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        print(f"🚀 Processing {len(requests)} requests...")
        
        # Process the batch with high-level interface
        results = await process_chat_completion_batch(limiter, requests)
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        # Show results
        for i, result in enumerate(successful):
            if result.result and result.result.choices:
                content = result.result.choices[0].message.content
                print(f"📝 Response {i+1}: {content}")


async def custom_requests_batch_example():
    """Example with custom ChatCompletionRequest objects."""
    print("\n🛠️ Custom Requests Batch Example")
    print("-" * 45)
    
    # Create custom requests with different parameters
    requests = [
        ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="Write a haiku about programming."),
            ],
            max_tokens=50,
            temperature=0.8
        ),
        ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a poetry expert."),
                Message(role=MessageRole.USER, content="Write a short limerick about Python."),
            ],
            max_tokens=60,
            temperature=0.9
        ),
        ChatCompletionRequest(
            model="gpt-4o",
            messages=[
                Message(role=MessageRole.USER, content="Explain quantum physics in simple terms."),
            ],
            max_tokens=100,
            temperature=0.3  # Lower temperature for factual content
        ),
    ]
    
    # Configure batch processing
    config = BatchConfig(
        max_concurrent_requests=2,
        max_retries_per_item=2,
        group_by_model=True,
    )
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        print(f"🚀 Processing {len(requests)} custom requests...")
        
        results = await process_chat_completion_batch(limiter, requests, config)
        
        # Show detailed results
        for i, result in enumerate(results):
            if result.success and result.result:
                response = result.result
                print(f"\n📝 Request {i+1}:")
                print(f"   Model: {response.model}")
                print(f"   Content: {response.choices[0].message.content if response.choices else 'No content'}")
                print(f"   Duration: {result.duration:.2f}s")
                if response.usage:
                    print(f"   Tokens: {response.usage.total_tokens}")
            else:
                print(f"\n❌ Request {i+1} failed: {result.error}")


async def multi_provider_batch_example():
    """Example processing batches across multiple providers."""
    print("\n🔄 Multi-Provider Batch Example")
    print("-" * 45)
    
    # Different requests for different providers
    openai_prompts = ["What is machine learning?", "Explain neural networks."]
    anthropic_prompts = ["What is deep learning?", "Explain transformers."]
    
    # OpenAI batch
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("🤖 Processing OpenAI batch...")
        openai_requests = create_chat_completion_requests(
            model="gpt-4o",
            prompts=openai_prompts,
            max_tokens=50
        )
        
        async with ChatLimiter.for_model("gpt-4o", openai_key) as limiter:
            openai_results = await process_chat_completion_batch(limiter, openai_requests)
            successful = [r for r in openai_results if r.success]
            print(f"✅ OpenAI: {len(successful)}/{len(openai_results)} successful")
    
    # Anthropic batch
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("🧠 Processing Anthropic batch...")
        anthropic_requests = create_chat_completion_requests(
            model="claude-3-5-sonnet-20241022",
            prompts=anthropic_prompts,
            max_tokens=50
        )
        
        async with ChatLimiter.for_model("claude-3-5-sonnet-20241022", anthropic_key) as limiter:
            anthropic_results = await process_chat_completion_batch(limiter, anthropic_requests)
            successful = [r for r in anthropic_results if r.success]
            print(f"✅ Anthropic: {len(successful)}/{len(anthropic_results)} successful")


def sync_batch_example():
    """Synchronous batch processing example."""
    print("\n🔄 Synchronous High-Level Batch Example")
    print("-" * 45)
    
    # Create simple requests
    requests = create_chat_completion_requests(
        model="gpt-4o",
        prompts=["Count to 3", "Count to 5", "Count to 7"],
        max_tokens=30
    )
    
    with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        print(f"🚀 Processing {len(requests)} requests synchronously...")
        
        # Process synchronously
        results = process_chat_completion_batch_sync(limiter, requests)
        
        successful = [r for r in results if r.success]
        print(f"✅ Completed {len(successful)} requests synchronously")
        
        # Show a sample result
        if successful and successful[0].result:
            content = successful[0].result.choices[0].message.content if successful[0].result.choices else ""
            print(f"📝 Sample response: {content}")


async def advanced_batch_configuration():
    """Example showing advanced batch configuration."""
    print("\n⚙️ Advanced Batch Configuration Example")
    print("-" * 45)
    
    # Create a larger batch
    prompts = [f"Tell me an interesting fact about the number {i}" for i in range(1, 11)]
    requests = create_chat_completion_requests(
        model="gpt-4o",
        prompts=prompts,
        max_tokens=40,
        temperature=0.8
    )
    
    # Advanced configuration
    config = BatchConfig(
        max_concurrent_requests=5,        # Process 5 at a time
        max_retries_per_item=3,          # Retry failed items up to 3 times
        retry_delay=2.0,                 # 2 second base retry delay
        stop_on_first_error=False,       # Continue processing even if some fail
        group_by_model=True,             # Group requests by model
        adaptive_batch_size=True,        # Adapt batch size to rate limits
    )
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        print(f"🚀 Processing {len(requests)} requests with advanced config...")
        
        results = await process_chat_completion_batch(limiter, requests, config)
        
        # Get statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"📊 Batch Statistics:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_duration = sum(r.duration for r in successful) / len(successful)
            avg_tokens = sum(r.result.usage.total_tokens for r in successful if r.result and r.result.usage) / len(successful)
            print(f"   Average duration: {avg_duration:.2f}s")
            print(f"   Average tokens: {avg_tokens:.0f}")


async def error_handling_batch_example():
    """Example showing error handling in batch processing."""
    print("\n🛡️ Batch Error Handling Example")
    print("-" * 45)
    
    # Mix of valid and problematic requests
    requests = [
        ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="This should work")],
            max_tokens=20
        ),
        ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="This should also work")],
            max_tokens=20
        ),
        # This might cause issues if the API key is invalid
        ChatCompletionRequest(
            model="invalid-model-name",
            messages=[Message(role=MessageRole.USER, content="This might fail")],
            max_tokens=20
        ),
    ]
    
    config = BatchConfig(
        max_retries_per_item=1,  # Minimal retries for demo
        stop_on_first_error=False,  # Continue despite errors
        collect_errors=True,
    )
    
    try:
        async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
            results = await process_chat_completion_batch(limiter, requests, config)
            
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"✅ Successful requests: {len(successful)}")
            print(f"❌ Failed requests: {len(failed)}")
            
            # Show error details
            for i, result in enumerate(failed):
                print(f"   Error {i+1}: {result.error}")
                
    except Exception as e:
        print(f"✅ Caught batch processing error: {type(e).__name__}")


async def comparison_example():
    """Example comparing low-level vs high-level batch processing."""
    print("\n🔍 Low-Level vs High-Level Comparison")
    print("-" * 45)
    
    prompts = ["Hello!", "How are you?", "What's the weather like?"]
    
    # High-level approach (recommended)
    print("🎯 High-level approach:")
    requests = create_chat_completion_requests(
        model="gpt-4o",
        prompts=prompts,
        max_tokens=20
    )
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        high_level_results = await process_chat_completion_batch(limiter, requests)
        print(f"   ✅ Processed {len([r for r in high_level_results if r.success])} requests")
        
        # Show first response using high-level objects
        if high_level_results and high_level_results[0].success and high_level_results[0].result:
            response = high_level_results[0].result
            print(f"   📝 Response: {response.choices[0].message.content if response.choices else 'No content'}")
            print(f"   📊 Usage: {response.usage}")
    
    print("\n💡 Benefits of high-level interface:")
    print("   - ✅ Type-safe request/response objects")
    print("   - ✅ Automatic provider format conversion")  
    print("   - ✅ Unified interface across providers")
    print("   - ✅ Built-in validation and error handling")
    print("   - ✅ Rich metadata and usage information")


async def main():
    """Run all high-level batch processing examples."""
    print("📦 Chat-Limiter High-Level Batch Processing Examples")
    print("=" * 65)
    print("These examples demonstrate batch processing with the new high-level")
    print("chat completion interface that works with typed request/response objects.\n")
    
    await simple_batch_example()
    await custom_requests_batch_example()
    await multi_provider_batch_example()
    sync_batch_example()
    await advanced_batch_configuration()
    await error_handling_batch_example()
    await comparison_example()
    
    print("\n✅ All high-level batch processing examples completed!")
    print("\n💡 Key Benefits of High-Level Batch Processing:")
    print("- ✅ Type-safe ChatCompletionRequest/Response objects")
    print("- ✅ Automatic provider format conversion")
    print("- ✅ Rich metadata and usage tracking")
    print("- ✅ Unified interface across all providers")
    print("- ✅ Built-in error handling and retry logic")
    print("- ✅ Flexible batch configuration options")
    print("- ✅ Both sync and async support")
    
    print("\n🔧 Environment Variables:")
    print("- OPENAI_API_KEY: For OpenAI models")
    print("- ANTHROPIC_API_KEY: For Anthropic models")
    print("- OPENROUTER_API_KEY: For OpenRouter models")


if __name__ == "__main__":
    asyncio.run(main())