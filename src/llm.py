import logging
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, HF_API_TOKEN, HF_MODEL

logger = logging.getLogger(__name__)

_llm = None
_llm_provider = None


def _try_ollama():
    """Try to connect to Ollama and return a ChatOllama instance."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=256,
            num_ctx=2048,
        )
        # Quick connectivity test
        llm.invoke("Hi")
        return llm
    except Exception:
        return None


def _try_huggingface():
    """Try to use HuggingFace Inference API as fallback."""
    if not HF_API_TOKEN:
        return None
    try:
        from langchain_huggingface import HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_API_TOKEN,
            temperature=0.7,
            max_new_tokens=512,
        )
        return llm
    except Exception:
        return None


def get_llm():
    """Get the LLM instance. Tries Ollama first, then HuggingFace."""
    global _llm, _llm_provider

    if _llm is not None:
        return _llm, _llm_provider

    # Try Ollama first
    logger.info(f"Attempting Ollama connection at {OLLAMA_BASE_URL}...")
    _llm = _try_ollama()
    if _llm:
        _llm_provider = "ollama"
        logger.info(f"Connected to Ollama (model: {OLLAMA_MODEL})")
        return _llm, _llm_provider

    # Try HuggingFace fallback
    logger.info("Ollama unavailable, trying HuggingFace fallback...")
    _llm = _try_huggingface()
    if _llm:
        _llm_provider = "huggingface"
        logger.info(f"Connected to HuggingFace (model: {HF_MODEL})")
        return _llm, _llm_provider

    logger.error("No LLM available")
    raise ConnectionError(
        "No LLM available. Please either:\n"
        "1. Install and run Ollama (https://ollama.com) with: ollama pull llama3.2\n"
        "2. Set HF_API_TOKEN in .env for HuggingFace fallback"
    )


def reset_llm():
    """Reset the LLM instance (for reconnection attempts)."""
    global _llm, _llm_provider
    _llm = None
    _llm_provider = None
