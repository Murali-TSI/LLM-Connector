"""
Async Basic Chat Completion Example

Demonstrates async chat completion with the OpenAI provider.
"""

import asyncio
from llm_connector import ConnectorFactory


async def main():
    # Create connector (uses OPENAI_API_KEY env var)
    connector = ConnectorFactory.create("openai")

    # Simple async string message
    print("=" * 50)
    print("Example 1: Async simple string message")
    print("=" * 50)
    
    response = await connector.async_chat().invoke(
        messages="What is the capital of France?"
    )
    
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print()

    # With custom parameters
    print("=" * 50)
    print("Example 2: Async with custom parameters")
    print("=" * 50)
    
    response = await connector.async_chat().invoke(
        messages="Write a haiku about Python programming",
        model="gpt-4o-mini",
        temperature=0.9,
        max_tokens=100,
    )
    
    print(f"Response: {response.content}")
    print()

    # Multi-turn conversation using structured messages
    print("=" * 50)
    print("Example 3: Async using structured messages")
    print("=" * 50)
    
    from llm_connector import (
        SystemMessage,
        UserMessage,
        TextBlock,
        Role,
    )

    messages = [
        SystemMessage(
            role=Role.SYSTEM,
            content=[TextBlock(text="You are a helpful assistant that speaks like a pirate.")]
        ),
        UserMessage(
            role=Role.USER,
            content=[TextBlock(text="Hello, how are you today?")]
        ),
    ]

    response = await connector.async_chat().invoke(messages=messages)
    print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
