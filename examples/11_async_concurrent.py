"""
Async Concurrent Requests Example

Demonstrates making multiple async requests concurrently for better performance.
"""

import asyncio
import time
from llm_connector import ConnectorFactory


async def main():
    connector = ConnectorFactory.create("openai")
    chat = connector.async_chat()

    print("=" * 50)
    print("Example 1: Sequential vs Concurrent requests")
    print("=" * 50)

    questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ]

    # Sequential execution (for comparison)
    print("\nSequential execution:")
    start = time.perf_counter()
    
    sequential_results = []
    for q in questions:
        response = await chat.invoke(messages=q, max_tokens=50)
        sequential_results.append(response.content)
    
    sequential_time = time.perf_counter() - start
    print(f"  Time: {sequential_time:.2f}s")
    for i, result in enumerate(sequential_results):
        print(f"  Q{i+1}: {result[:50]}...")

    # Concurrent execution
    print("\nConcurrent execution:")
    start = time.perf_counter()
    
    tasks = [
        chat.invoke(messages=q, max_tokens=50)
        for q in questions
    ]
    concurrent_results = await asyncio.gather(*tasks)
    
    concurrent_time = time.perf_counter() - start
    print(f"  Time: {concurrent_time:.2f}s")
    print(f"  Speedup: {sequential_time / concurrent_time:.1f}x faster")
    for i, result in enumerate(concurrent_results):
        print(f"  Q{i+1}: {result.content[:50]}...")

    print()
    print("=" * 50)
    print("Example 2: Concurrent requests with error handling")
    print("=" * 50)

    async def safe_invoke(message: str, index: int):
        """Wrapper that handles errors gracefully."""
        try:
            response = await chat.invoke(messages=message, max_tokens=100)
            return {"index": index, "success": True, "content": response.content}
        except Exception as e:
            return {"index": index, "success": False, "error": str(e)}

    messages = [
        "Write a one-line joke about cats",
        "Write a one-line joke about dogs",
        "Write a one-line joke about birds",
    ]

    tasks = [safe_invoke(msg, i) for i, msg in enumerate(messages)]
    results = await asyncio.gather(*tasks)

    print("\nResults:")
    for result in results:
        if result["success"]:
            print(f"  [{result['index']}] ✓ {result['content'][:60]}...")
        else:
            print(f"  [{result['index']}] ✗ Error: {result['error']}")

    print()
    print("=" * 50)
    print("Example 3: Concurrent streaming requests")
    print("=" * 50)

    async def stream_with_label(message: str, label: str):
        """Stream a response and collect it with a label."""
        stream = await chat.invoke(messages=message, stream=True, max_tokens=100)
        content = []
        async for chunk in stream:
            if chunk.delta_content:
                content.append(chunk.delta_content)
        return {"label": label, "content": "".join(content)}

    prompts = [
        ("Write a haiku about morning", "Morning Haiku"),
        ("Write a haiku about evening", "Evening Haiku"),
        ("Write a haiku about night", "Night Haiku"),
    ]

    tasks = [stream_with_label(msg, label) for msg, label in prompts]
    results = await asyncio.gather(*tasks)

    print("\nConcurrent streaming results:")
    for result in results:
        print(f"\n{result['label']}:")
        print(f"  {result['content']}")

    print()
    print("=" * 50)
    print("Example 4: Rate-limited concurrent requests")
    print("=" * 50)

    async def rate_limited_requests(messages: list, max_concurrent: int = 3):
        """Execute requests with a concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_invoke(msg: str, idx: int):
            async with semaphore:
                print(f"  Starting request {idx}...")
                response = await chat.invoke(messages=msg, max_tokens=30)
                print(f"  Completed request {idx}")
                return response.content
        
        tasks = [limited_invoke(msg, i) for i, msg in enumerate(messages)]
        return await asyncio.gather(*tasks)

    messages = [f"What is {i} + {i}?" for i in range(1, 7)]
    
    print(f"\nExecuting {len(messages)} requests with max 3 concurrent:")
    results = await rate_limited_requests(messages, max_concurrent=3)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  {i}: {result[:40]}...")


if __name__ == "__main__":
    asyncio.run(main())
