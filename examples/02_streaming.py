"""
Streaming Response Example

Demonstrates streaming chat completions for real-time output.
"""

from llm_connector import ConnectorFactory


def main():
    connector = ConnectorFactory.create("openai")

    # Basic streaming
    print("=" * 50)
    print("Example 1: Basic streaming")
    print("=" * 50)

    stream = connector.chat().invoke(
        messages="Write a short story about a robot learning to paint (3 paragraphs)",
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.delta_content:
            print(chunk.delta_content, end="", flush=True)
            full_response += chunk.delta_content

        # Check for finish reason on last chunk
        if chunk.finish_reason:
            print(f"\n\n[Finished: {chunk.finish_reason}]")

        # Usage is available on the last chunk
        if chunk.usage:
            print(f"[Tokens: {chunk.usage.total_tokens}]")

    print()

    # Streaming with progress indicator
    print("=" * 50)
    print("Example 2: Streaming with token counter")
    print("=" * 50)

    stream = connector.chat().invoke(
        messages="List 5 interesting facts about the ocean",
        stream=True,
    )

    token_count = 0
    for chunk in stream:
        if chunk.delta_content:
            print(chunk.delta_content, end="", flush=True)
            # Rough token estimate (actual tokens in usage at end)
            token_count += 1

    print(f"\n\n[Approximate chunks received: {token_count}]")
    print()

    # Collecting streamed response
    print("=" * 50)
    print("Example 3: Collecting full response from stream")
    print("=" * 50)

    stream = connector.chat().invoke(
        messages="What are the primary colors?",
        stream=True,
    )

    collected_content = []
    final_usage = None

    for chunk in stream:
        if chunk.delta_content:
            collected_content.append(chunk.delta_content)
        if chunk.usage:
            final_usage = chunk.usage

    full_response = "".join(collected_content)
    print(f"Full response: {full_response}")

    if final_usage:
        print(f"Prompt tokens: {final_usage.prompt_tokens}")
        print(f"Completion tokens: {final_usage.completion_tokens}")
        print(f"Total tokens: {final_usage.total_tokens}")


if __name__ == "__main__":
    main()
