"""
LLM Connector Library

A unified interface for interacting with multiple LLM providers.

Usage:
    from llm_connector import ConnectorFactory

    # Create a connector
    connector = ConnectorFactory.create("openai", config={"api_key": "..."})

    # Sync chat completion
    response = connector.chat().invoke(messages="Hello, how are you?")
    print(response.content)

    # Sync streaming
    for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
        print(chunk.delta_content, end="", flush=True)

    # Async chat completion
    async def main():
        response = await connector.async_chat().invoke(messages="Hello!")
        print(response.content)

        # Async streaming
        async for chunk in await connector.async_chat().invoke(messages="Tell me a story", stream=True):
            print(chunk.delta_content, end="", flush=True)
"""

from .base import (
    # LLMConnector
    LLMConnector,
    # Message types
    Role,
    TextBlock,
    ImageBlock,
    DocumentBlock,
    ContentBlock,
    ToolCall,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Message,
    Conversation,
    # Completion (sync)
    ChatCompletion,
    ChatResponses,
    ChatStreamChunks,
    Usage,
    ToolCallDelta,
    # Completion (async)
    AsyncChatCompletion,
    # Batch (sync)
    BatchStatus,
    BatchTimestamp,
    BatchRequest,
    BatchResult,
    BatchProcess,
    # Batch (async)
    AsyncBatchProcess,
    # File (sync)
    PurposeType,
    FileObject,
    FileAPI,
    # File (async)
    AsyncFileAPI,
)

from .factory import ConnectorFactory

from .exceptions import (
    ProviderNotSupportedError,
    ProviderImportError,
    AuthenticationError,
    RateLimitError,
    APIError,
    InvalidRequestError,
    ContentFilterError,
    ContextLengthExceededError,
    BatchError,
    FileError,
)


__all__ = [
    # Factory
    "ConnectorFactory",
    # LLMConnector
    "LLMConnector",
    # Message types
    "Role",
    "TextBlock",
    "ImageBlock",
    "DocumentBlock",
    "ContentBlock",
    "ToolCall",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "Message",
    "Conversation",
    # Completion (sync)
    "ChatCompletion",
    "ChatResponses",
    "ChatStreamChunks",
    "Usage",
    "ToolCallDelta",
    # Completion (async)
    "AsyncChatCompletion",
    # Batch (sync)
    "BatchStatus",
    "BatchTimestamp",
    "BatchRequest",
    "BatchResult",
    "BatchProcess",
    # Batch (async)
    "AsyncBatchProcess",
    # File (sync)
    "PurposeType",
    "FileObject",
    "FileAPI",
    # File (async)
    "AsyncFileAPI",
    # Exceptions
    "ProviderNotSupportedError",
    "ProviderImportError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "InvalidRequestError",
    "ContentFilterError",
    "ContextLengthExceededError",
    "BatchError",
    "FileError",
]

__version__ = "0.1.0"
