import pytest
from unittest.mock import MagicMock, patch, mock_open

from llm_connector.base import BatchStatus
from llm_connector.providers.groq.batch import GroqBatchProcess
from llm_connector.exceptions import BatchError


class TestGroqBatchProcess:
    """Tests for GroqBatchProcess."""

    def test_create_batch_from_file_path(self, sample_batch_response):
        """Test creating batch from file path."""
        mock_client = MagicMock()
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response
        mock_client.batches.create.return_value = sample_batch_response

        batch = GroqBatchProcess(mock_client)

        with patch("builtins.open", mock_open(read_data=b'{"test": "data"}')):
            result = batch.create(file="test.jsonl")

        assert result.id == "batch_123"
        assert result.status == BatchStatus.COMPLETED

    def test_create_batch_from_bytes(self, sample_batch_response):
        """Test creating batch from bytes."""
        mock_client = MagicMock()
        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response
        mock_client.batches.create.return_value = sample_batch_response

        batch = GroqBatchProcess(mock_client)
        result = batch.create(file=b'{"test": "data"}')

        assert result.id == "batch_123"
        mock_client.files.create.assert_called_once()

    def test_status(self, sample_batch_response):
        """Test getting batch status."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = sample_batch_response

        batch = GroqBatchProcess(mock_client)
        result = batch.status("batch_123")

        assert result.id == "batch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.batches.retrieve.assert_called_with("batch_123")

    def test_result_completed(self, sample_batch_response):
        """Test getting results of completed batch."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = sample_batch_response

        mock_content = MagicMock()
        mock_content.text = (
            '{"id": "1", "result": "success"}\n{"id": "2", "result": "success"}'
        )
        mock_client.files.content.return_value = mock_content

        batch = GroqBatchProcess(mock_client)
        result = batch.result("batch_123")

        assert result.job_id == "batch_123"
        assert result.output_file_id == "file-output-456"
        assert len(result.records) == 2

    def test_result_not_completed(self):
        """Test getting results of non-completed batch raises error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_response

        batch = GroqBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            batch.result("batch_123")
        assert "not completed" in str(exc_info.value)

    def test_cancel(self, sample_batch_response):
        """Test cancelling a batch."""
        mock_client = MagicMock()
        sample_batch_response.status = "cancelled"
        mock_client.batches.cancel.return_value = sample_batch_response

        batch = GroqBatchProcess(mock_client)
        result = batch.cancel("batch_123")

        assert result.status == BatchStatus.CANCELLED
        mock_client.batches.cancel.assert_called_with("batch_123")

    def test_list(self, sample_batch_response):
        """Test listing batches."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_batch_response]
        mock_client.batches.list.return_value = mock_response

        batch = GroqBatchProcess(mock_client)
        results = batch.list(limit=10)

        assert len(results) == 1
        assert results[0].id == "batch_123"

    def test_list_with_pagination(self, sample_batch_response):
        """Test listing batches with pagination."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_batch_response]
        mock_client.batches.list.return_value = mock_response

        batch = GroqBatchProcess(mock_client)
        batch.list(limit=10, after="batch_122")

        mock_client.batches.list.assert_called_with(limit=10, after="batch_122")

    def test_batch_request_timestamps(self, sample_batch_response):
        """Test batch request timestamps are converted correctly."""
        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = sample_batch_response

        batch = GroqBatchProcess(mock_client)
        result = batch.status("batch_123")

        assert result.timestamps.created_at is not None
        assert result.timestamps.completed_at is not None

    def test_batch_status_mapping(self):
        """Test all batch statuses are mapped correctly."""
        mock_client = MagicMock()
        batch = GroqBatchProcess(mock_client)

        statuses = [
            "validating",
            "failed",
            "in_progress",
            "finalizing",
            "completed",
            "expired",
            "cancelled",
            "cancelling",
        ]

        for status in statuses:
            mock_response = MagicMock()
            mock_response.id = "batch_123"
            mock_response.status = status
            mock_response.created_at = 1700000000
            mock_response.in_progress_at = None
            mock_response.cancelled_at = None
            mock_response.completed_at = None
            mock_response.expired_at = None
            mock_response.failed_at = None
            mock_response.finalizing_at = None
            mock_response.completion_window = "24h"
            mock_response.input_file_id = "file-123"
            mock_response.output_file_id = None
            mock_response.error_file_id = None
            mock_response.endpoint = "/v1/chat/completions"
            mock_response.request_counts = None

            result = batch._to_batch_request(mock_response)
            assert result.status is not None


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
