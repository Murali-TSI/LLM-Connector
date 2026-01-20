"""
LLM Connector Library

A unified interface for interacting with multiple LLM providers.

Usage:
    from llm import ConnectorFactory

    # Create a connector
    connector = ConnectorFactory.create("openai", config={"api_key": "..."})

    # Chat completion
    response = connector.chat().invoke(messages="Hello, how are you?")
    print(response.content)

    # Streaming
    for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
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
    # Completion
    ChatCompletion,
    ChatResponses,
    ChatStreamChunks,
    Usage,
    ToolCallDelta,
    # Batch
    BatchStatus,
    BatchTimestamp,
    BatchRequest,
    BatchResult,
    BatchProcess,
    # File
    PurposeType,
    FileObject,
    FileAPI,
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
    # Completion
    "ChatCompletion",
    "ChatResponses",
    "ChatStreamChunks",
    "Usage",
    "ToolCallDelta",
    # Batch
    "BatchStatus",
    "BatchTimestamp",
    "BatchRequest",
    "BatchResult",
    "BatchProcess",
    # File
    "PurposeType",
    "FileObject",
    "FileAPI",
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
