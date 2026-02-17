import os
from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# HuggingFace fallback
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Text splitter
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# RAG
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

# Paths
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")
