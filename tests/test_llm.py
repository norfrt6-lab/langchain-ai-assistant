from unittest.mock import patch, MagicMock
import pytest
import src.llm as llm_module


def setup_function():
    """Reset LLM singleton before each test."""
    llm_module._llm = None
    llm_module._llm_provider = None


def test_ollama_connection_success():
    """Test successful Ollama connection."""
    with patch("src.llm._try_ollama") as mock_ollama:
        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm

        llm, provider = llm_module.get_llm()

        assert provider == "ollama"
        assert llm == mock_llm


def test_fallback_to_huggingface():
    """Test fallback to HuggingFace when Ollama fails."""
    with patch("src.llm._try_ollama", return_value=None), \
         patch("src.llm._try_huggingface") as mock_hf:
        mock_llm = MagicMock()
        mock_hf.return_value = mock_llm

        llm, provider = llm_module.get_llm()

        assert provider == "huggingface"
        assert llm == mock_llm


def test_connection_error_when_both_fail():
    """Test ConnectionError when both Ollama and HuggingFace fail."""
    with patch("src.llm._try_ollama", return_value=None), \
         patch("src.llm._try_huggingface", return_value=None):
        with pytest.raises(ConnectionError):
            llm_module.get_llm()


def test_singleton_behavior():
    """Test that get_llm returns cached instance on second call."""
    with patch("src.llm._try_ollama") as mock_ollama:
        mock_llm = MagicMock()
        mock_ollama.return_value = mock_llm

        llm1, _ = llm_module.get_llm()
        llm2, _ = llm_module.get_llm()

        assert llm1 is llm2
        assert mock_ollama.call_count == 1


def test_reset_llm():
    """Test that reset_llm clears the cached instance."""
    llm_module._llm = MagicMock()
    llm_module._llm_provider = "ollama"

    llm_module.reset_llm()

    assert llm_module._llm is None
    assert llm_module._llm_provider is None
