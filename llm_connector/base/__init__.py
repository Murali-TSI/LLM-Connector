from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .file import PurposeType, FileObject, FileAPI, AsyncFileAPI
from .batch import (
    BatchStatus,
    BatchTimestamp,
    BatchRequest,
    BatchResult,
    BatchProcess,
    AsyncBatchProcess,
)
from .completion import (
    ChatCompletion,
    AsyncChatCompletion,
    ChatResponses,
    ChatStreamChunks,
    Usage,
    ToolCallDelta,
)
from .message import (
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
)


class LLMConnector(ABC):
    """Abstract base class for LLM provider connectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration. Override in subclasses."""
        return None

    # Sync methods
    @abstractmethod
    def chat(self) -> ChatCompletion:
        """Get the chat completion interface."""
        pass

    @abstractmethod
    def batch(self) -> BatchProcess:
        """Get the batch processing interface."""
        pass

    @abstractmethod
    def file(self) -> FileAPI:
        """Get the file API interface."""
        pass

    # Async methods
    @abstractmethod
    def async_chat(self) -> AsyncChatCompletion:
        """Get the async chat completion interface."""
        pass

    @abstractmethod
    def async_batch(self) -> AsyncBatchProcess:
        """Get the async batch processing interface."""
        pass

    @abstractmethod
    def async_file(self) -> AsyncFileAPI:
        """Get the async file API interface."""
        pass


__all__ = [
    # Message
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
    # LLMConnector
    "LLMConnector",
]
