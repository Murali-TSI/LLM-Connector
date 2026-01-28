from __future__ import annotations

import json
from typing import Union, Any, Optional, Literal, List, BinaryIO, TYPE_CHECKING

from ...exceptions import BatchError, AuthenticationError, APIError
from ...base import (
    BatchProcess,
    AsyncBatchProcess,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BatchTimestamp,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic import AsyncAnthropic


class AnthropicBatchProcess(BatchProcess):
    """
    Anthropic Message Batches API implementation.

    Note: Anthropic's batch API works differently from OpenAI/Groq:
    - Requests are submitted directly as a list, not via file uploads
    - The `file` parameter accepts JSONL content (bytes/str) or a list of request dicts
    - Each request must have a `custom_id` and `params` with Messages API parameters

    Example JSONL format:
        {"custom_id": "req-1", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}}
        {"custom_id": "req-2", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hi"}]}}

    Or pass requests directly:
        batch.create(requests=[
            {"custom_id": "req-1", "params": {"model": "...", "max_tokens": 1024, "messages": [...]}},
            {"custom_id": "req-2", "params": {"model": "...", "max_tokens": 1024, "messages": [...]}},
        ])
    """

    def __init__(self, client: "Anthropic") -> None:
        self._client = client

    def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO, None] = None,
        completion_window: Literal["24h"] = "24h",
        requests: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> BatchRequest:
        """
        Create a new message batch.

        Args:
            file: JSONL file (path, bytes, or file-like object) with batch requests.
                  Each line should be: {"custom_id": "...", "params": {...}}
            completion_window: Time window for completion (only "24h" supported, for interface compatibility)
            requests: Alternative to file - list of request dicts directly
            **kwargs: Additional arguments

        Returns:
            BatchRequest with job details

        Note:
            Either `file` or `requests` must be provided.
            If using `file`, it should contain JSONL with Anthropic-format requests.
        """
        try:
            batch_requests = self._parse_requests(file, requests)

            if not batch_requests:
                raise BatchError(
                    "No requests provided. Use 'file' or 'requests' parameter."
                )

            response = self._client.messages.batches.create(requests=batch_requests)
            return self._to_batch_request(response)

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Get the status of a message batch.

        Args:
            job_id: The batch ID (e.g., "msgbatch_...")

        Returns:
            BatchRequest with current status
        """
        try:
            response = self._client.messages.batches.retrieve(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """
        Get the results of a completed message batch.

        Args:
            job_id: The batch ID

        Returns:
            BatchResult with output records

        Note:
            Results are only available after processing_status is "ended".
            Each record contains custom_id and result (succeeded/errored/canceled/expired).
        """
        try:
            batch = self._client.messages.batches.retrieve(job_id)

            if batch.processing_status != "ended":
                raise BatchError(
                    f"Batch job is not completed. Current status: {batch.processing_status}"
                )

            records = []
            for entry in self._client.messages.batches.results(job_id):
                record = {
                    "custom_id": entry.custom_id,
                    "result": self._format_result_entry(entry.result),
                }
                records.append(record)

            return BatchResult(
                job_id=job_id,
                output_file_id=None,  # Anthropic doesn't use file IDs
                records=records,
            )

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Cancel a message batch.

        Args:
            job_id: The batch ID

        Returns:
            BatchRequest with updated status

        Note:
            Cancellation is not immediate. The batch enters "canceling" state
            and may complete some in-progress requests before fully canceling.
        """
        try:
            response = self._client.messages.batches.cancel(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """
        List message batches.

        Args:
            limit: Maximum number of batches to return (1-100)
            after: Cursor for pagination (batch ID to start after)

        Returns:
            List of BatchRequest objects
        """
        try:
            list_kwargs = {"limit": limit}
            if after:
                list_kwargs["after_id"] = after

            batches = []
            for batch in self._client.messages.batches.list(**list_kwargs):
                batches.append(self._to_batch_request(batch))
                if len(batches) >= limit:
                    break

            return batches

        except Exception as e:
            raise self._handle_exception(e)

    def _parse_requests(
        self,
        file: Union[str, bytes, BinaryIO, None],
        requests: Optional[List[dict]],
    ) -> List[dict]:
        """Parse batch requests from file or direct list."""
        if requests is not None:
            return requests

        if file is None:
            return []

        if isinstance(file, str):
            with open(file, "r") as f:
                content = f.read()
        elif isinstance(file, bytes):
            content = file.decode("utf-8")
        else:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")

        parsed_requests = []
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    parsed_requests.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise BatchError(f"Invalid JSON in batch file: {e}")

        return parsed_requests

    def _format_result_entry(self, result) -> dict:
        """Format a batch result entry."""
        result_dict = {"type": result.type}

        if result.type == "succeeded":
            if hasattr(result, "message"):
                msg = result.message
                result_dict["message"] = {
                    "id": msg.id,
                    "type": msg.type,
                    "role": msg.role,
                    "content": [
                        self._format_content_block(block) for block in msg.content
                    ],
                    "model": msg.model,
                    "stop_reason": msg.stop_reason,
                    "usage": (
                        {
                            "input_tokens": msg.usage.input_tokens,
                            "output_tokens": msg.usage.output_tokens,
                        }
                        if msg.usage
                        else None
                    ),
                }
        elif result.type == "errored":
            if hasattr(result, "error"):
                result_dict["error"] = {
                    "type": result.error.type,
                    "message": result.error.message,
                }

        return result_dict

    def _format_content_block(self, block) -> dict:
        """Format a content block from the response."""
        if hasattr(block, "type"):
            if block.type == "text":
                return {"type": "text", "text": block.text}
            elif block.type == "tool_use":
                return {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
        return {"type": "unknown"}

    def _to_batch_request(self, response) -> BatchRequest:
        """Convert Anthropic batch response to BatchRequest."""
        status_map = {
            "in_progress": BatchStatus.IN_PROGRESS,
            "canceling": BatchStatus.CANCELLED,
            "ended": BatchStatus.COMPLETED,
        }

        # Build timestamps
        timestamps = BatchTimestamp(
            created_at=response.created_at if response.created_at else "",
            in_progress_at=None,  # Anthropic doesn't provide this
            cancelled_at=response.cancel_initiated_at,
            completed_at=response.ended_at,
            expired_at=response.expires_at,
            failed_at=None,  # Anthropic doesn't have a separate failed state
            finalized_at=response.ended_at,
        )

        request_counts = None
        if response.request_counts:
            request_counts = {
                "total": (
                    response.request_counts.processing
                    + response.request_counts.succeeded
                    + response.request_counts.errored
                    + response.request_counts.canceled
                    + response.request_counts.expired
                ),
                "completed": response.request_counts.succeeded,
                "failed": response.request_counts.errored,
                "canceled": response.request_counts.canceled,
                "expired": response.request_counts.expired,
                "processing": response.request_counts.processing,
            }

        return BatchRequest(
            id=response.id,
            status=status_map.get(response.processing_status, BatchStatus.IN_PROGRESS),
            timestamps=timestamps,
            completion_window="24h",
            input_file_id="",  # Anthropic doesn't use file IDs
            output_file_id=None,
            error_file_id=None,
            endpoint="/v1/messages",
            request_counts=request_counts,
        )

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return BatchError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, anthropic.NotFoundError):
            return BatchError(f"Batch not found: {e}")
        elif isinstance(e, anthropic.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return BatchError(str(e))


class AnthropicAsyncBatchProcess(AsyncBatchProcess):
    """
    Anthropic Async Message Batches API implementation.

    See AnthropicBatchProcess for details on usage.
    """

    def __init__(self, client: "AsyncAnthropic") -> None:
        self._client = client

    async def create(
        self,
        *,
        file: Union[str, bytes, BinaryIO, None] = None,
        completion_window: Literal["24h"] = "24h",
        requests: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> BatchRequest:
        """
        Create a new message batch asynchronously.

        Args:
            file: JSONL file (path, bytes, or file-like object) with batch requests
            completion_window: Time window for completion (only "24h" supported)
            requests: Alternative to file - list of request dicts directly
            **kwargs: Additional arguments

        Returns:
            BatchRequest with job details
        """
        try:
            batch_requests = self._parse_requests(file, requests)

            if not batch_requests:
                raise BatchError(
                    "No requests provided. Use 'file' or 'requests' parameter."
                )

            response = await self._client.messages.batches.create(
                requests=batch_requests
            )
            return self._to_batch_request(response)

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    async def status(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Get the status of a message batch asynchronously.

        Args:
            job_id: The batch ID

        Returns:
            BatchRequest with current status
        """
        try:
            response = await self._client.messages.batches.retrieve(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def result(self, job_id: str, **kwargs: Any) -> BatchResult:
        """
        Get the results of a completed message batch asynchronously.

        Args:
            job_id: The batch ID

        Returns:
            BatchResult with output records
        """
        try:
            batch = await self._client.messages.batches.retrieve(job_id)

            if batch.processing_status != "ended":
                raise BatchError(
                    f"Batch job is not completed. Current status: {batch.processing_status}"
                )

            records = []
            async for entry in await self._client.messages.batches.results(job_id):
                record = {
                    "custom_id": entry.custom_id,
                    "result": self._format_result_entry(entry.result),
                }
                records.append(record)

            return BatchResult(
                job_id=job_id,
                output_file_id=None,
                records=records,
            )

        except BatchError:
            raise
        except Exception as e:
            raise self._handle_exception(e)

    async def cancel(self, job_id: str, **kwargs: Any) -> BatchRequest:
        """
        Cancel a message batch asynchronously.

        Args:
            job_id: The batch ID

        Returns:
            BatchRequest with updated status
        """
        try:
            response = await self._client.messages.batches.cancel(job_id)
            return self._to_batch_request(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def list(
        self, *, limit: int = 20, after: Optional[str] = None, **kwargs: Any
    ) -> List[BatchRequest]:
        """
        List message batches asynchronously.

        Args:
            limit: Maximum number of batches to return
            after: Cursor for pagination

        Returns:
            List of BatchRequest objects
        """
        try:
            list_kwargs = {"limit": limit}
            if after:
                list_kwargs["after_id"] = after

            batches = []
            async for batch in self._client.messages.batches.list(**list_kwargs):
                batches.append(self._to_batch_request(batch))
                if len(batches) >= limit:
                    break

            return batches

        except Exception as e:
            raise self._handle_exception(e)

    def _parse_requests(
        self,
        file: Union[str, bytes, BinaryIO, None],
        requests: Optional[List[dict]],
    ) -> List[dict]:
        """Parse batch requests from file or direct list."""
        if requests is not None:
            return requests

        if file is None:
            return []

        if isinstance(file, str):
            with open(file, "r") as f:
                content = f.read()
        elif isinstance(file, bytes):
            content = file.decode("utf-8")
        else:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")

        parsed_requests = []
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    parsed_requests.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise BatchError(f"Invalid JSON in batch file: {e}")

        return parsed_requests

    def _format_result_entry(self, result) -> dict:
        """Format a batch result entry."""
        result_dict = {"type": result.type}

        if result.type == "succeeded":
            if hasattr(result, "message"):
                msg = result.message
                result_dict["message"] = {
                    "id": msg.id,
                    "type": msg.type,
                    "role": msg.role,
                    "content": [
                        self._format_content_block(block) for block in msg.content
                    ],
                    "model": msg.model,
                    "stop_reason": msg.stop_reason,
                    "usage": (
                        {
                            "input_tokens": msg.usage.input_tokens,
                            "output_tokens": msg.usage.output_tokens,
                        }
                        if msg.usage
                        else None
                    ),
                }
        elif result.type == "errored":
            if hasattr(result, "error"):
                result_dict["error"] = {
                    "type": result.error.type,
                    "message": result.error.message,
                }

        return result_dict

    def _format_content_block(self, block) -> dict:
        """Format a content block from the response."""
        if hasattr(block, "type"):
            if block.type == "text":
                return {"type": "text", "text": block.text}
            elif block.type == "tool_use":
                return {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
        return {"type": "unknown"}

    def _to_batch_request(self, response) -> BatchRequest:
        """Convert Anthropic batch response to BatchRequest."""
        status_map = {
            "in_progress": BatchStatus.IN_PROGRESS,
            "canceling": BatchStatus.CANCELLED,
            "ended": BatchStatus.COMPLETED,
        }

        timestamps = BatchTimestamp(
            created_at=response.created_at if response.created_at else "",
            in_progress_at=None,
            cancelled_at=response.cancel_initiated_at,
            completed_at=response.ended_at,
            expired_at=response.expires_at,
            failed_at=None,
            finalized_at=response.ended_at,
        )

        request_counts = None
        if response.request_counts:
            request_counts = {
                "total": (
                    response.request_counts.processing
                    + response.request_counts.succeeded
                    + response.request_counts.errored
                    + response.request_counts.canceled
                    + response.request_counts.expired
                ),
                "completed": response.request_counts.succeeded,
                "failed": response.request_counts.errored,
                "canceled": response.request_counts.canceled,
                "expired": response.request_counts.expired,
                "processing": response.request_counts.processing,
            }

        return BatchRequest(
            id=response.id,
            status=status_map.get(response.processing_status, BatchStatus.IN_PROGRESS),
            timestamps=timestamps,
            completion_window="24h",
            input_file_id="",
            output_file_id=None,
            error_file_id=None,
            endpoint="/v1/messages",
            request_counts=request_counts,
        )

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return BatchError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, anthropic.NotFoundError):
            return BatchError(f"Batch not found: {e}")
        elif isinstance(e, anthropic.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return BatchError(str(e))
