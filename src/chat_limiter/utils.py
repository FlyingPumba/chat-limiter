"""
Utilities for handling asyncio execution contexts.

Fail-fast, minimal helpers to bridge async code into sync callers.
"""

import asyncio
import concurrent.futures
from typing import Any, TypeVar

T = TypeVar("T")


def run_coro_blocking(coro: Any) -> T:
    """Run the given coroutine to completion, regardless of event loop state.

    - If no loop is running, call asyncio.run(coro).
    - If a loop is running (e.g., Jupyter), execute asyncio.run(coro) in a new thread.
    """
    try:
        asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        in_async_context = False

    if not in_async_context:
        try:
            return asyncio.run(coro)  # type: ignore[no-any-return]
        finally:
            # Ensure the coroutine is closed even if asyncio.run was mocked
            # and did not consume it, to avoid "coroutine was never awaited" warnings.
            try:
                coro.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _runner() -> T:
        return asyncio.run(coro)  # type: ignore[no-any-return]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future: concurrent.futures.Future[T] = executor.submit(_runner)
            return future.result()
        finally:
            # Close the coroutine in case the background runner did not
            # actually execute it (e.g., asyncio.run mocked in tests).
            try:
                coro.close()  # type: ignore[attr-defined]
            except Exception:
                pass


