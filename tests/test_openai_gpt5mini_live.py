"""
Live test to verify external usage pattern against openai/gpt-5-mini.

Assumptions:
- OPENAI_API_KEY is set in the environment.
- Dynamic model discovery will find "gpt-5-mini" when using provider-prefixed
  model id "openai/gpt-5-mini".
"""

import os
from dataclasses import dataclass
from typing import Any

import pytest

from chat_limiter import (
    BatchConfig,
    ChatCompletionRequest,
    ChatLimiter,
    Message,
    MessageRole,
    process_chat_completion_batch,
)


@pytest.mark.skipif(not bool(os.getenv("OPENAI_API_KEY")), reason="OPENAI_API_KEY not set")
class TestOpenAIGPT5MiniLive:
    """Live test replicating external library usage for gpt-5-mini."""

    @dataclass
    class LLM:
        llm_model_name: str
        provider: str | None = None
        api_key: str | None = None
        chat_timeout: float = 240.0
        max_tokens: int = 100
        temperature: float = 0.0
        max_concurrent: int = 4

        async def _generate_batch_llm_response(
            self, key_to_messages: dict[Any, list[Message]]
        ) -> dict[Any, str | None]:
            assert isinstance(key_to_messages, dict) and len(key_to_messages) > 0

            keys = list(key_to_messages.keys())

            requests: list[ChatCompletionRequest] = []
            for key in keys:
                requests.append(
                    ChatCompletionRequest(
                        model=self.llm_model_name,
                        messages=key_to_messages[key],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                )

            config = BatchConfig(
                max_concurrent_requests=int(self.max_concurrent),
                max_retries_per_item=0,
                show_progress=False,
                # print_prompts=True,
                # print_responses=True,
                # verbose_exceptions=True,
            )

            async with ChatLimiter.for_model(
                self.llm_model_name,
                api_key=self.api_key,
                timeout=self.chat_timeout,
                provider=self.provider,
            ) as limiter:
                results = await process_chat_completion_batch(limiter, requests, config)

            key_to_response: dict[Any, str | None] = {}
            for i, result in enumerate(results):
                if result.success and result.result and result.result.choices:
                    key_to_response[keys[i]] = result.result.choices[0].message.content
                else:
                    key_to_response[keys[i]] = None

            assert len(key_to_response) == len(keys)
            return key_to_response

    @pytest.mark.asyncio
    async def test_external_usage_pattern_sanity(self):
        model = "openai/gpt-5-mini"
        api_key = os.getenv("OPENAI_API_KEY")

        llm = self.LLM(
            llm_model_name=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=100,
            max_concurrent=1,
        )

        key_to_messages = {
            "check": [
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="Reply with just the word OK."),
            ]
        }

        results = await llm._generate_batch_llm_response(key_to_messages)

        assert isinstance(results, dict) and "check" in results
        text = results["check"]
        assert isinstance(text, str) and text.strip() != ""


