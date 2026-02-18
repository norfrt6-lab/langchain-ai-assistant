import os
import importlib


def test_default_values():
    """Test that config provides sensible defaults."""
    import src.config as config

    assert config.OLLAMA_BASE_URL == os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    assert config.OLLAMA_MODEL == os.getenv("OLLAMA_MODEL", "llama3.2")
    assert config.EMBEDDING_MODEL == os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    assert isinstance(config.CHUNK_SIZE, int)
    assert isinstance(config.CHUNK_OVERLAP, int)
    assert isinstance(config.TOP_K_RESULTS, int)
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP >= 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_env_override(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("TOP_K_RESULTS", "10")

    import src.config as config
    importlib.reload(config)

    assert config.OLLAMA_MODEL == "test-model"
    assert config.CHUNK_SIZE == 500
    assert config.TOP_K_RESULTS == 10
