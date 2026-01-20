import pytest
from unittest.mock import MagicMock, patch
import os


class TestOpenAIConnector:
    """Tests for OpenAIConnector."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-env-key"})
    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_env_key(self, mock_openai):
        """Test initialization with environment variable API key."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector()
        
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "test-env-key"

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_config_key(self, mock_openai):
        """Test initialization with config API key."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector(config={"api_key": "config-key"})
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "config-key"

    @patch.dict(os.environ, {}, clear=True)
    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_without_key_raises(self, mock_openai):
        """Test initialization without API key raises error."""
        from llm_connector.providers.openai import OpenAIConnector
        from llm_connector.exceptions import AuthenticationError
        
        # Remove OPENAI_API_KEY if it exists
        os.environ.pop("OPENAI_API_KEY", None)
        
        with pytest.raises(AuthenticationError):
            OpenAIConnector()

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_base_url(self, mock_openai):
        """Test initialization with custom base URL."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        OpenAIConnector(config={
            "api_key": "test-key",
            "base_url": "https://custom.api.com"
        })
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "https://custom.api.com"

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_organization(self, mock_openai):
        """Test initialization with organization."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        OpenAIConnector(config={
            "api_key": "test-key",
            "organization": "org-123"
        })
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["organization"] == "org-123"

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_timeout(self, mock_openai):
        """Test initialization with timeout."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        OpenAIConnector(config={
            "api_key": "test-key",
            "timeout": 30
        })
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["timeout"] == 30

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_init_with_max_retries(self, mock_openai):
        """Test initialization with max retries."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        OpenAIConnector(config={
            "api_key": "test-key",
            "max_retries": 5
        })
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["max_retries"] == 5

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", False)
    def test_init_without_openai_package(self):
        """Test initialization without openai package raises error."""
        # Need to reload the module to pick up the patched value
        import importlib
        import llm_connector.providers.openai as openai_module
        
        # Store original value
        original_available = openai_module.OPENAI_AVAILABLE
        openai_module.OPENAI_AVAILABLE = False
        
        try:
            from llm_connector.exceptions import ProviderImportError
            
            with pytest.raises(ProviderImportError):
                openai_module.OpenAIConnector(config={"api_key": "test"})
        finally:
            openai_module.OPENAI_AVAILABLE = original_available

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_chat_returns_completion(self, mock_openai):
        """Test chat() returns ChatCompletion instance."""
        from llm_connector.providers.openai import OpenAIConnector
        from llm_connector.base import ChatCompletion
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector(config={"api_key": "test-key"})
        chat = connector.chat()
        
        assert isinstance(chat, ChatCompletion)

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_chat_is_cached(self, mock_openai):
        """Test chat() returns same instance on multiple calls."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector(config={"api_key": "test-key"})
        chat1 = connector.chat()
        chat2 = connector.chat()
        
        assert chat1 is chat2

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_batch_returns_batch_process(self, mock_openai):
        """Test batch() returns BatchProcess instance."""
        from llm_connector.providers.openai import OpenAIConnector
        from llm_connector.base import BatchProcess
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector(config={"api_key": "test-key"})
        batch = connector.batch()
        
        assert isinstance(batch, BatchProcess)

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_file_returns_file_api(self, mock_openai):
        """Test file() returns FileAPI instance."""
        from llm_connector.providers.openai import OpenAIConnector
        from llm_connector.base import FileAPI
        
        mock_openai.return_value = MagicMock()
        
        connector = OpenAIConnector(config={"api_key": "test-key"})
        file_api = connector.file()
        
        assert isinstance(file_api, FileAPI)

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_client_property(self, mock_openai):
        """Test client property returns underlying client."""
        from llm_connector.providers.openai import OpenAIConnector
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        connector = OpenAIConnector(config={"api_key": "test-key"})
        
        assert connector.client is mock_client
