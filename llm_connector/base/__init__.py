from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .file import PurposeType, FileObject, FileAPI
from .batch import BatchStatus, BatchTimestamp, BatchRequest, BatchResult, BatchProcess
from .completion import ChatCompletion, ChatResponses, ChatStreamChunks, Usage, ToolCallDelta
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
    # LLMConnector
    "LLMConnector",
]
