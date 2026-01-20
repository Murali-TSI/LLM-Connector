from __future__ import annotations

from typing import Union, Optional, BinaryIO, List, TYPE_CHECKING

from ...base import FileAPI, FileObject, PurposeType
from ...exceptions import FileError, AuthenticationError, APIError

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIFileAPI(FileAPI):
    """OpenAI Files API implementation."""

    def __init__(self, client: "OpenAI") -> None:
        self._client = client

    def upload(self, *, file: Union[str, bytes, BinaryIO], purpose: PurposeType) -> str:
        """
        Upload a file to OpenAI.

        Args:
            file: File path, bytes, or file-like object
            purpose: Purpose of the file ('fine-tune' or 'batch')

        Returns:
            The file ID
        """
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    response = self._client.files.create(file=f, purpose=purpose)
            elif isinstance(file, bytes):
                response = self._client.files.create(file=("file.jsonl", file), purpose=purpose)
            else:
                response = self._client.files.create(file=file, purpose=purpose)

            return response.id

        except Exception as e:
            raise self._handle_exception(e)

    def retrieve(self, *, file_id: str) -> FileObject:
        """
        Retrieve file metadata.

        Args:
            file_id: The ID of the file

        Returns:
            FileObject with file metadata
        """
        try:
            response = self._client.files.retrieve(file_id)
            return FileObject(
                id=response.id,
                filename=response.filename,
                purpose=response.purpose,
                bytes=response.bytes,
                created_at=response.created_at,
                status=response.status,
            )
        except Exception as e:
            raise self._handle_exception(e)

    def download(self, *, file_id: str) -> bytes:
        """
        Download file content.

        Args:
            file_id: The ID of the file

        Returns:
            File content as bytes
        """
        try:
            response = self._client.files.content(file_id)
            return response.content
        except Exception as e:
            raise self._handle_exception(e)

    def delete(self, *, file_id: str) -> None:
        """
        Delete a file.

        Args:
            file_id: The ID of the file to delete
        """
        try:
            self._client.files.delete(file_id)
        except Exception as e:
            raise self._handle_exception(e)

    def list(self, *, purpose: Optional[PurposeType] = None) -> List[FileObject]:
        """
        List files.

        Args:
            purpose: Optional filter by purpose

        Returns:
            List of FileObject instances
        """
        try:
            kwargs = {}
            if purpose:
                kwargs["purpose"] = purpose

            response = self._client.files.list(**kwargs)

            return [
                FileObject(
                    id=f.id,
                    filename=f.filename,
                    purpose=f.purpose,
                    bytes=f.bytes,
                    created_at=f.created_at,
                    status=f.status,
                ) for f in response.data
            ]
        except Exception as e:
            raise self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert OpenAI exceptions to our custom exceptions."""
        try:
            import openai
        except ImportError:
            return FileError(str(e))

        if isinstance(e, openai.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, openai.NotFoundError):
            return FileError(f"File not found: {e}")
        elif isinstance(e, openai.APIError):
            return APIError(str(e), status_code=getattr(e, "status_code", None))
        else:
            return FileError(str(e))
