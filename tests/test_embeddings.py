from unittest.mock import patch, MagicMock
import src.embeddings as embeddings_module


def test_get_embeddings_returns_instance():
    """Test that get_embeddings returns an embeddings instance."""
    embeddings_module._embeddings = None

    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
        mock_instance = MagicMock()
        mock_hf.return_value = mock_instance

        result = embeddings_module.get_embeddings()

        assert result == mock_instance
        mock_hf.assert_called_once()


def test_get_embeddings_singleton():
    """Test that get_embeddings returns the same instance on repeated calls."""
    embeddings_module._embeddings = None

    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
        mock_instance = MagicMock()
        mock_hf.return_value = mock_instance

        result1 = embeddings_module.get_embeddings()
        result2 = embeddings_module.get_embeddings()

        assert result1 is result2
        assert mock_hf.call_count == 1


def test_get_embeddings_uses_config():
    """Test that embeddings use the configured model name."""
    embeddings_module._embeddings = None

    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf, \
         patch("src.embeddings.EMBEDDING_MODEL", "test-model"):
        embeddings_module.get_embeddings()

        call_kwargs = mock_hf.call_args
        assert call_kwargs[1]["model_name"] == "test-model"
