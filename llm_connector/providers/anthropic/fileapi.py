from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, BinaryIO, List, TYPE_CHECKING

from ...base import FileAPI, AsyncFileAPI, FileObject, PurposeType
from ...exceptions import FileError, AuthenticationError, APIError

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic import AsyncAnthropic

# Beta header required for Files API
FILES_API_BETA = "files-api-2025-04-14"


class AnthropicFileAPI(FileAPI):
    """
    Anthropic Beta Files API implementation.

    The Files API lets you upload and manage files to use with the Claude API
    without re-uploading content with each request. This is useful for:
    - Referencing documents/images across multiple API calls
    - Providing inputs for code execution tool
    - Downloading outputs from code execution/skills

    Note: This is a beta feature requiring the files-api-2025-04-14 beta header.

    Example:
        ```python
        # Upload a file
        file_id = connector.file().upload(file="document.pdf", purpose="user_data")

        # Use in messages
        response = connector.chat().create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this document"},
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id}
                    }
                ]
            }]
        )
        ```
    """

    def __init__(self, client: "Anthropic") -> None:
        self._client = client

    def upload(self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType) -> str:
        """
        Upload a file to Anthropic's Files API.

        Args:
            file: File path, bytes, or file-like object to upload
            purpose: Purpose of the file (for interface compatibility, not used by Anthropic)

        Returns:
            The file ID (e.g., "file_011CNha8iCJcU1wXNR6q4V8w")

        Note:
            Anthropic doesn't use the `purpose` parameter like OpenAI does.
            Files are automatically categorized based on their MIME type.
        """
        try:
            if isinstance(file, str):
                file_path = Path(file)
                response = self._client.beta.files.upload(
                    file=file_path,
                    betas=[FILES_API_BETA],
                )
            elif isinstance(file, bytes):
                import io

                file_obj = io.BytesIO(file)
                response = self._client.beta.files.upload(
                    file=file_obj,
                    betas=[FILES_API_BETA],
                )
            else:
                response = self._client.beta.files.upload(
                    file=file,
                    betas=[FILES_API_BETA],
                )

            return response.id

        except Exception as e:
            raise self._handle_exception(e)

    def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata.

        Args:
            file_id: The file ID to retrieve

        Returns:
            FileObject with file metadata
        """
        try:
            response = self._client.beta.files.retrieve_metadata(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )
            return self._to_file_object(response)

        except Exception as e:
            raise self._handle_exception(e)

    def download(self, *, file_id: str) -> bytes:
        """
        Download file content.

        Args:
            file_id: The file ID to download

        Returns:
            File content as bytes

        Note:
            You can only download files that were created by skills or the
            code execution tool. Files that you uploaded cannot be downloaded.
        """
        try:
            response = self._client.beta.files.download(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )
            return response.read()

        except Exception as e:
            raise self._handle_exception(e)

    def delete(self, *, file_id: str) -> None:
        """
        Delete a file.

        Args:
            file_id: The file ID to delete

        Note:
            Files are inaccessible via the API shortly after deletion,
            but may persist in active Messages API calls.
        """
        try:
            self._client.beta.files.delete(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )

        except Exception as e:
            raise self._handle_exception(e)

    def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files in the workspace.

        Args:
            purpose: Filter by purpose (not used by Anthropic, for interface compatibility)

        Returns:
            List of FileObject with file metadata

        Note:
            Anthropic doesn't support filtering by purpose. All files in the
            workspace are returned.
        """
        try:
            files = []
            for file_metadata in self._client.beta.files.list(betas=[FILES_API_BETA]):
                files.append(self._to_file_object(file_metadata))

            return files

        except Exception as e:
            raise self._handle_exception(e)

    def _to_file_object(self, response) -> FileObject:
        """Convert Anthropic FileMetadata to FileObject."""
        return FileObject(
            id=response.id,
            bytes=response.size_bytes,
            created_at=response.created_at,
            filename=response.filename,
            purpose="user_data",  # Anthropic doesn't have purpose concept
            status="processed",  # Files are immediately available
            status_details=None,
        )

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return FileError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, anthropic.NotFoundError):
            return FileError(f"File not found: {e}")
        elif isinstance(e, anthropic.BadRequestError):
            return FileError(f"Invalid request: {e}")
        elif isinstance(e, anthropic.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return FileError(str(e))


class AnthropicAsyncFileAPI(AsyncFileAPI):
    """
    Anthropic Async Beta Files API implementation.

    See AnthropicFileAPI for details on usage.
    """

    def __init__(self, client: "AsyncAnthropic") -> None:
        self._client = client

    async def upload(
        self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType
    ) -> str:
        """
        Upload a file to Anthropic's Files API asynchronously.

        Args:
            file: File path, bytes, or file-like object to upload
            purpose: Purpose of the file (for interface compatibility)

        Returns:
            The file ID
        """
        try:
            if isinstance(file, str):
                file_path = Path(file)
                response = await self._client.beta.files.upload(
                    file=file_path,
                    betas=[FILES_API_BETA],
                )
            elif isinstance(file, bytes):
                import io

                file_obj = io.BytesIO(file)
                response = await self._client.beta.files.upload(
                    file=file_obj,
                    betas=[FILES_API_BETA],
                )
            else:
                response = await self._client.beta.files.upload(
                    file=file,
                    betas=[FILES_API_BETA],
                )

            return response.id

        except Exception as e:
            raise self._handle_exception(e)

    async def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata asynchronously.

        Args:
            file_id: The file ID to retrieve

        Returns:
            FileObject with file metadata
        """
        try:
            response = await self._client.beta.files.retrieve_metadata(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )
            return self._to_file_object(response)

        except Exception as e:
            raise self._handle_exception(e)

    async def download(self, *, file_id: str) -> bytes:
        """
        Download file content asynchronously.

        Args:
            file_id: The file ID to download

        Returns:
            File content as bytes

        Note:
            Only files created by skills/code execution can be downloaded.
        """
        try:
            response = await self._client.beta.files.download(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )
            return await response.read()

        except Exception as e:
            raise self._handle_exception(e)

    async def delete(self, *, file_id: str) -> None:
        """
        Delete a file asynchronously.

        Args:
            file_id: The file ID to delete
        """
        try:
            await self._client.beta.files.delete(
                file_id=file_id,
                betas=[FILES_API_BETA],
            )

        except Exception as e:
            raise self._handle_exception(e)

    async def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files in the workspace asynchronously.

        Args:
            purpose: Filter by purpose (not used by Anthropic)

        Returns:
            List of FileObject with file metadata
        """
        try:
            files = []
            async for file_metadata in self._client.beta.files.list(
                betas=[FILES_API_BETA]
            ):
                files.append(self._to_file_object(file_metadata))

            return files

        except Exception as e:
            raise self._handle_exception(e)

    def _to_file_object(self, response) -> FileObject:
        """Convert Anthropic FileMetadata to FileObject."""
        return FileObject(
            id=response.id,
            bytes=response.size_bytes,
            created_at=response.created_at,
            filename=response.filename,
            purpose="user_data",
            status="processed",
            status_details=None,
        )

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return FileError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, anthropic.NotFoundError):
            return FileError(f"File not found: {e}")
        elif isinstance(e, anthropic.BadRequestError):
            return FileError(f"Invalid request: {e}")
        elif isinstance(e, anthropic.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return FileError(str(e))
