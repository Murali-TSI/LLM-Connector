import pytest
from unittest.mock import MagicMock, patch
import os


class TestGroqConnector:
    """Tests for GroqConnector."""

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-env-key"})
    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_with_env_key(self, mock_groq):
        """Test initialization with environment variable API key."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector()
        
        mock_groq.assert_called_once()
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs["api_key"] == "test-env-key"

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_with_config_key(self, mock_groq):
        """Test initialization with config API key."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "config-key"})
        
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs["api_key"] == "config-key"

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_without_key_raises(self, mock_groq):
        """Test initialization without API key raises error."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.exceptions import AuthenticationError
        
        # Remove GROQ_API_KEY if it exists
        os.environ.pop("GROQ_API_KEY", None)
        
        with pytest.raises(AuthenticationError):
            GroqConnector()

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_with_base_url(self, mock_groq):
        """Test initialization with custom base URL."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        GroqConnector(config={
            "api_key": "test-key",
            "base_url": "https://custom.api.com"
        })
        
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.com"

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_with_timeout(self, mock_groq):
        """Test initialization with timeout."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        GroqConnector(config={
            "api_key": "test-key",
            "timeout": 30
        })
        
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs["timeout"] == 30

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_init_with_max_retries(self, mock_groq):
        """Test initialization with max retries."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        GroqConnector(config={
            "api_key": "test-key",
            "max_retries": 5
        })
        
        call_kwargs = mock_groq.call_args.kwargs
        assert call_kwargs["max_retries"] == 5

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", False)
    def test_init_without_groq_package(self):
        """Test initialization without groq package raises error."""
        import importlib
        import llm_connector.providers.groq as groq_module
        
        original_available = groq_module.GROQ_AVAILABLE
        groq_module.GROQ_AVAILABLE = False
        
        try:
            from llm_connector.exceptions import ProviderImportError
            
            with pytest.raises(ProviderImportError):
                groq_module.GroqConnector(config={"api_key": "test"})
        finally:
            groq_module.GROQ_AVAILABLE = original_available

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_chat_returns_completion(self, mock_groq):
        """Test chat() returns ChatCompletion instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import ChatCompletion
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        chat = connector.chat()
        
        assert isinstance(chat, ChatCompletion)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_chat_is_cached(self, mock_groq):
        """Test chat() returns same instance on multiple calls."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        chat1 = connector.chat()
        chat2 = connector.chat()
        
        assert chat1 is chat2

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_batch_returns_batch_process(self, mock_groq):
        """Test batch() returns BatchProcess instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import BatchProcess
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        batch = connector.batch()
        
        assert isinstance(batch, BatchProcess)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_file_returns_file_api(self, mock_groq):
        """Test file() returns FileAPI instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import FileAPI
        
        mock_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        file_api = connector.file()
        
        assert isinstance(file_api, FileAPI)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.Groq")
    def test_client_property(self, mock_groq):
        """Test client property returns underlying client."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        connector = GroqConnector(config={"api_key": "test-key"})
        
        assert connector.client is mock_client

    # ==================== Async Tests ====================

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_chat_returns_async_completion(self, mock_groq, mock_async_groq):
        """Test async_chat() returns AsyncChatCompletion instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import AsyncChatCompletion
        
        mock_groq.return_value = MagicMock()
        mock_async_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        async_chat = connector.async_chat()
        
        assert isinstance(async_chat, AsyncChatCompletion)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_chat_is_cached(self, mock_groq, mock_async_groq):
        """Test async_chat() returns same instance on multiple calls."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        mock_async_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        async_chat1 = connector.async_chat()
        async_chat2 = connector.async_chat()
        
        assert async_chat1 is async_chat2

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_batch_returns_async_batch_process(self, mock_groq, mock_async_groq):
        """Test async_batch() returns AsyncBatchProcess instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import AsyncBatchProcess
        
        mock_groq.return_value = MagicMock()
        mock_async_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        async_batch = connector.async_batch()
        
        assert isinstance(async_batch, AsyncBatchProcess)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_file_returns_async_file_api(self, mock_groq, mock_async_groq):
        """Test async_file() returns AsyncFileAPI instance."""
        from llm_connector.providers.groq import GroqConnector
        from llm_connector.base import AsyncFileAPI
        
        mock_groq.return_value = MagicMock()
        mock_async_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        async_file = connector.async_file()
        
        assert isinstance(async_file, AsyncFileAPI)

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_client_lazy_initialization(self, mock_groq, mock_async_groq):
        """Test async client is lazily initialized."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        mock_async_client = MagicMock()
        mock_async_groq.return_value = mock_async_client
        
        connector = GroqConnector(config={"api_key": "test-key"})
        
        # AsyncGroq should not be called yet
        mock_async_groq.assert_not_called()
        
        # Access async_client property
        client = connector.async_client
        
        # Now it should be called
        mock_async_groq.assert_called_once()
        assert client is mock_async_client

    @patch("llm_connector.providers.groq.GROQ_AVAILABLE", True)
    @patch("llm_connector.providers.groq.AsyncGroq")
    @patch("llm_connector.providers.groq.Groq")
    def test_async_client_reuses_same_instance(self, mock_groq, mock_async_groq):
        """Test async client returns same instance on multiple accesses."""
        from llm_connector.providers.groq import GroqConnector
        
        mock_groq.return_value = MagicMock()
        mock_async_groq.return_value = MagicMock()
        
        connector = GroqConnector(config={"api_key": "test-key"})
        
        client1 = connector.async_client
        client2 = connector.async_client
        
        # Should only be called once
        assert mock_async_groq.call_count == 1
        assert client1 is client2
