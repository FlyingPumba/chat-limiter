"""
Live test that sends a small batch request to the OpenAI model using a
provider-prefixed id for detection ("openai/gpt-4o").

Assumption: OpenAI API expects base model ids (e.g., "gpt-4o"), so we use the
prefixed id only for provider detection and pass the base id to the request.
"""

import os
import pytest

from chat_limiter import ChatLimiter
from chat_limiter.batch import (
    BatchConfig,
    create_chat_completion_requests,
    process_chat_completion_batch,
)


@pytest.mark.skipif(not bool(os.getenv("OPENAI_API_KEY")), reason="OPENAI_API_KEY not set")
class TestOpenAIGPT4oLive:
    """Live tests for sending a small request to openai/gpt-4o via batch API."""

    @pytest.mark.asyncio
    async def test_small_request_model_has_provider_prefix(self):
        model = "openai/gpt-4o"

        prompts = ["Say 'hi' in one word."]
        max_tokens = 5
        temperature = 0.0

        requests = create_chat_completion_requests(
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        config = BatchConfig(
            max_concurrent_requests=1,
            max_retries_per_item=0,
            group_by_model=True,
            json_mode=False,
            show_progress=False,
            print_prompts=False,
            print_responses=False,
        )

        async with ChatLimiter.for_model(model, timeout=240.0) as limiter:
            results = await process_chat_completion_batch(limiter, requests, config)

        assert len(results) == 1
        first = results[0]
        assert first.success, f"Request failed: {first.error_message}"
        assert first.result is not None

        response = first.result
        assert response.provider == "openai"
        assert response.choices and len(response.choices) > 0
        content = response.choices[0].message.content
        assert content is not None

