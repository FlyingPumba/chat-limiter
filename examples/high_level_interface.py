"""
High-level interface examples for chat-limiter.

This example demonstrates the new high-level chat completion interface that
provides a clean, SDK-like experience similar to OpenAI and Anthropic clients.
"""

import asyncio
import os
from typing import List

from chat_limiter import ChatLimiter, Message, MessageRole


async def simple_chat_example():
    """Simple chat completion example using the high-level interface."""
    print("🤖 Simple Chat Example")
    print("-" * 40)
    
    # Automatic provider detection from model name
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        # Simple single prompt
        response = await limiter.simple_chat(
            model="gpt-4o",
            prompt="What is the capital of France?",
            max_tokens=50
        )
        print(f"✅ Simple response: {response}")


async def conversation_example():
    """Multi-turn conversation example."""
    print("\n💬 Conversation Example")
    print("-" * 40)
    
    # Build a conversation with multiple messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello! Can you help me with Python?"),
        Message(role=MessageRole.ASSISTANT, content="Of course! I'd be happy to help you with Python. What would you like to know?"),
        Message(role=MessageRole.USER, content="How do I create a list comprehension?"),
    ]
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        response = await limiter.chat_completion(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        if response.choices:
            print(f"✅ Assistant: {response.choices[0].message.content}")
            print(f"📊 Usage: {response.usage}")


async def multi_provider_example():
    """Example using different providers."""
    print("\n🔄 Multi-Provider Example")
    print("-" * 40)
    
    # OpenAI example
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        async with ChatLimiter.for_model("gpt-4o", openai_key) as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o",
                prompt="Explain machine learning in one sentence.",
                max_tokens=50
            )
            print(f"🤖 OpenAI: {response}")
    
    # Anthropic example
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        async with ChatLimiter.for_model("claude-3-5-sonnet-20241022", anthropic_key) as limiter:
            response = await limiter.simple_chat(
                model="claude-3-5-sonnet-20241022",
                prompt="Explain machine learning in one sentence.",
                max_tokens=50
            )
            print(f"🧠 Anthropic: {response}")
    
    # OpenRouter example
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        async with ChatLimiter.for_model("openai/gpt-4o", openrouter_key) as limiter:
            response = await limiter.simple_chat(
                model="openai/gpt-4o",
                prompt="Explain machine learning in one sentence.",
                max_tokens=50
            )
            print(f"🔀 OpenRouter: {response}")


def sync_example():
    """Synchronous example using the high-level interface."""
    print("\n🔄 Synchronous Example")
    print("-" * 40)
    
    # Synchronous usage
    with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        response = limiter.simple_chat_sync(
            model="gpt-4o",
            prompt="What is Python?",
            max_tokens=50
        )
        print(f"✅ Sync response: {response}")


async def detailed_response_example():
    """Example showing detailed response information."""
    print("\n📊 Detailed Response Example")
    print("-" * 40)
    
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        messages = [
            Message(role=MessageRole.USER, content="Write a short poem about Python programming.")
        ]
        
        response = await limiter.chat_completion(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.8
        )
        
        print(f"📝 Response ID: {response.id}")
        print(f"🤖 Model: {response.model}")
        print(f"🏢 Provider: {response.provider}")
        print(f"⏰ Created: {response.created}")
        
        if response.choices:
            choice = response.choices[0]
            print(f"📍 Choice {choice.index}: {choice.message.content}")
            print(f"🔚 Finish reason: {choice.finish_reason}")
        
        if response.usage:
            usage = response.usage
            print(f"📊 Token usage:")
            print(f"   Prompt tokens: {usage.prompt_tokens}")
            print(f"   Completion tokens: {usage.completion_tokens}")
            print(f"   Total tokens: {usage.total_tokens}")


async def error_handling_example():
    """Example of error handling with the high-level interface."""
    print("\n🛡️ Error Handling Example")
    print("-" * 40)
    
    try:
        # This will fail with invalid API key
        async with ChatLimiter.for_model("gpt-4o", "invalid-key") as limiter:
            response = await limiter.simple_chat(
                model="gpt-4o",
                prompt="This will fail",
                max_tokens=10
            )
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}")
        print(f"📝 Error demonstrates proper error handling")


async def provider_specific_features():
    """Example showing provider-specific features."""
    print("\n⚙️ Provider-Specific Features")
    print("-" * 40)
    
    # OpenAI-specific parameters
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        async with ChatLimiter.for_model("gpt-4o", openai_key) as limiter:
            response = await limiter.chat_completion(
                model="gpt-4o",
                messages=[Message(role=MessageRole.USER, content="Tell me a joke")],
                max_tokens=50,
                temperature=0.9,
                frequency_penalty=0.5,  # OpenAI-specific
                presence_penalty=0.3,   # OpenAI-specific
            )
            print(f"🤖 OpenAI with penalties: {response.choices[0].message.content if response.choices else 'No response'}")
    
    # Anthropic-specific parameters
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        async with ChatLimiter.for_model("claude-3-5-sonnet-20241022", anthropic_key) as limiter:
            response = await limiter.chat_completion(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role=MessageRole.USER, content="Tell me a joke")],
                max_tokens=50,
                temperature=0.9,
                top_k=40,  # Anthropic-specific
            )
            print(f"🧠 Anthropic with top_k: {response.choices[0].message.content if response.choices else 'No response'}")


async def batch_chat_completions():
    """Example of batch processing with high-level interface."""
    print("\n📦 Batch Chat Completions")
    print("-" * 40)
    
    # Create multiple chat completion requests
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
        "What is Go?",
        "What is TypeScript?"
    ]
    
    # Note: This would need the batch processor to be updated to work with
    # the new high-level interface. For now, we'll simulate it:
    async with ChatLimiter.for_model("gpt-4o", os.getenv("OPENAI_API_KEY", "demo-key")) as limiter:
        print(f"🚀 Processing {len(questions)} questions...")
        
        # Process sequentially for demo (batch processing will be updated later)
        responses = []
        for i, question in enumerate(questions):
            try:
                response = await limiter.simple_chat(
                    model="gpt-4o",
                    prompt=question,
                    max_tokens=30
                )
                responses.append(response)
                print(f"✅ Question {i+1}: {response[:50]}...")
            except Exception as e:
                print(f"❌ Question {i+1} failed: {e}")
        
        print(f"📊 Completed {len(responses)} out of {len(questions)} questions")


async def main():
    """Run all high-level interface examples."""
    print("🚀 Chat-Limiter High-Level Interface Examples")
    print("=" * 60)
    print("These examples demonstrate the new high-level chat completion interface")
    print("that provides a clean, SDK-like experience similar to OpenAI and Anthropic clients.\n")
    
    await simple_chat_example()
    await conversation_example()
    await multi_provider_example()
    sync_example()
    await detailed_response_example()
    await error_handling_example()
    await provider_specific_features()
    await batch_chat_completions()
    
    print("\n✅ All high-level interface examples completed!")
    print("\n💡 Key Benefits of the High-Level Interface:")
    print("- ✅ Automatic provider detection from model names")
    print("- ✅ Unified Message and Response types across providers")
    print("- ✅ Clean, SDK-like API similar to OpenAI/Anthropic clients")
    print("- ✅ Provider-specific parameter support")
    print("- ✅ Automatic request/response format conversion")
    print("- ✅ Both sync and async support")
    print("- ✅ Built-in rate limiting and retry logic")
    
    print("\n🔧 Environment Variables:")
    print("- OPENAI_API_KEY: For OpenAI models")
    print("- ANTHROPIC_API_KEY: For Anthropic models")
    print("- OPENROUTER_API_KEY: For OpenRouter models")


if __name__ == "__main__":
    asyncio.run(main())