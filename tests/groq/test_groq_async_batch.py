import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

from llm_connector.base import BatchStatus
from llm_connector.providers.groq.batch import GroqAsyncBatchProcess
from llm_connector.exceptions import BatchError


class TestGroqAsyncBatchProcess:
    """Tests for GroqAsyncBatchProcess."""

    @pytest.mark.asyncio
    async def test_create_batch_from_bytes(self, sample_batch_response):
        """Test creating batch from bytes asynchronously."""
        mock_client = MagicMock()
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_client.files.create = AsyncMock(return_value=mock_file_response)
        mock_client.batches.create = AsyncMock(return_value=sample_batch_response)

        batch = GroqAsyncBatchProcess(mock_client)
        result = await batch.create(file=b'{"test": "data"}')

        assert result.id == "batch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.files.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_status(self, sample_batch_response):
        """Test getting batch status asynchronously."""
        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(return_value=sample_batch_response)

        batch = GroqAsyncBatchProcess(mock_client)
        result = await batch.status("batch_123")

        assert result.id == "batch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.batches.retrieve.assert_called_with("batch_123")

    @pytest.mark.asyncio
    async def test_result_completed(self, sample_batch_response):
        """Test getting results of completed batch asynchronously."""
        mock_client = MagicMock()
        mock_client.batches.retrieve = AsyncMock(return_value=sample_batch_response)

        mock_content = MagicMock()
        mock_content.text = (
            '{"id": "1", "result": "success"}\n{"id": "2", "result": "success"}'
        )
        mock_client.files.content = AsyncMock(return_value=mock_content)

        batch = GroqAsyncBatchProcess(mock_client)
        result = await batch.result("batch_123")

        assert result.job_id == "batch_123"
        assert result.output_file_id == "file-output-456"
        assert len(result.records) == 2

    @pytest.mark.asyncio
    async def test_result_not_completed(self):
        """Test getting results of non-completed batch raises error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "in_progress"
        mock_client.batches.retrieve = AsyncMock(return_value=mock_response)

        batch = GroqAsyncBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            await batch.result("batch_123")
        assert "not completed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel(self, sample_batch_response):
        """Test cancelling a batch asynchronously."""
        mock_client = MagicMock()
        sample_batch_response.status = "cancelled"
        mock_client.batches.cancel = AsyncMock(return_value=sample_batch_response)

        batch = GroqAsyncBatchProcess(mock_client)
        result = await batch.cancel("batch_123")

        assert result.status == BatchStatus.CANCELLED
        mock_client.batches.cancel.assert_called_with("batch_123")

    @pytest.mark.asyncio
    async def test_list(self, sample_batch_response):
        """Test listing batches asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_batch_response]
        mock_client.batches.list = AsyncMock(return_value=mock_response)

        batch = GroqAsyncBatchProcess(mock_client)
        results = await batch.list(limit=10)

        assert len(results) == 1
        assert results[0].id == "batch_123"

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, sample_batch_response):
        """Test listing batches with pagination asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_batch_response]
        mock_client.batches.list = AsyncMock(return_value=mock_response)

        batch = GroqAsyncBatchProcess(mock_client)
        await batch.list(limit=10, after="batch_122")

        mock_client.batches.list.assert_called_with(limit=10, after="batch_122")


@pytest.fixture
def sample_batch_response():
    """Create a sample Groq batch response."""
    response = MagicMock()
    response.id = "batch_123"
    response.status = "completed"
    response.created_at = 1700000000
    response.in_progress_at = 1700000100
    response.completed_at = 1700000200
    response.cancelled_at = None
    response.expired_at = None
    response.failed_at = None
    response.finalizing_at = 1700000150
    response.completion_window = "24h"
    response.input_file_id = "file-input-123"
    response.output_file_id = "file-output-456"
    response.error_file_id = None
    response.endpoint = "/v1/chat/completions"
    response.request_counts = MagicMock()
    response.request_counts.model_dump.return_value = {
        "total": 10,
        "completed": 10,
        "failed": 0,
    }
    return response
