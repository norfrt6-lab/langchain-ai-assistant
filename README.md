# RAG AI Assistant

A Retrieval-Augmented Generation (RAG) AI Assistant built with LangChain, ChromaDB, and Streamlit. Upload your documents (PDF, TXT, or web pages) and chat with them using a local, free LLM.

**Senior Project 2026**

## Features

- **Multi-format document support**: PDF, TXT files, and web pages
- **Local & free**: Uses Ollama for local LLM inference (no API costs)
- **Persistent storage**: ChromaDB stores document embeddings across sessions
- **Streaming chat**: Real-time response streaming with conversation history
- **Source attribution**: See which document chunks were used to generate each answer
- **RAG evaluation metrics**: Retrieval relevance scoring, response time tracking, per-chunk analysis
- **Evaluation dashboard**: Monitor RAG quality across all queries
- **Auto-fallback**: Falls back to HuggingFace API if Ollama is unavailable
- **Docker deployment**: One-command deployment with docker-compose
- **Unit tested**: 30+ tests with pytest covering all modules
- **Professional UI**: Custom styled interface with dark theme

## Architecture

```
Document (PDF/TXT/URL)
    |
    v
Document Loader (PyPDF, TextLoader, WebBaseLoader)
    |
    v
Text Splitter (RecursiveCharacterTextSplitter)
    |
    v
Embeddings (all-MiniLM-L6-v2 - HuggingFace)
    |
    v
Vector Store (ChromaDB - persistent, local)
    |
    v
User Question --> Similarity Search --> Top-K Chunks
    |
    v
LLM (Ollama / HuggingFace) --> Streaming Answer + Sources + Metrics
```

## Prerequisites

- **Python 3.10+**
- **Ollama** (recommended) - for free local LLM
- **Docker** (optional) - for containerized deployment

### Installing Ollama

1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Install and run Ollama
3. Pull a model:

```bash
# Recommended (8 GB RAM)
ollama pull llama3.2

# Lighter alternative (4 GB RAM)
ollama pull tinyllama

# Better quality (16 GB RAM)
ollama pull mistral
```

## Installation

### Option 1: Local Setup

1. Clone the repository:

```bash
git clone https://github.com/norfrt6-lab/langchain-ai-assistant.git
cd langchain-ai-assistant
```

2. Create a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment:

```bash
cp .env.example .env
```

5. Run:

```bash
streamlit run app.py
```

### Option 2: Docker

```bash
docker-compose up --build
```

This starts both the Streamlit app (port 8501) and Ollama (port 11434). Then pull a model:

```bash
docker exec -it ollama ollama pull llama3.2
```

Open `http://localhost:8501` in your browser.

## Running Tests

```bash
python -m pytest tests/ -v
```

All 30 tests cover: config, document loading, text splitting, embeddings, vector store, LLM, and RAG chain.

## Project Structure

```
langchain-ai-assistant/
├── app.py                    # Streamlit web application (Chat + Evaluation tabs)
├── Dockerfile                # Container build configuration
├── docker-compose.yml        # Multi-container deployment (app + Ollama)
├── requirements.txt          # Python dependencies
├── pytest.ini                # Test configuration
├── .env.example              # Environment configuration template
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized settings
│   ├── document_loader.py    # PDF, TXT, Web document loaders
│   ├── text_splitter.py      # Text chunking logic
│   ├── embeddings.py         # HuggingFace embedding model
│   ├── vector_store.py       # ChromaDB vector store operations
│   ├── llm.py                # LLM setup (Ollama + HuggingFace fallback)
│   ├── rag_chain.py          # RAG pipeline chain with streaming
│   ├── evaluation.py         # RAG quality metrics and evaluation
│   └── styles.py             # Custom CSS styling
├── tests/
│   ├── test_config.py        # Configuration tests
│   ├── test_document_loader.py # Document loader tests
│   ├── test_text_splitter.py # Text splitting tests
│   ├── test_embeddings.py    # Embedding model tests
│   ├── test_vector_store.py  # Vector store tests
│   ├── test_llm.py           # LLM connection tests
│   └── test_rag_chain.py     # RAG pipeline tests
├── data/                     # Uploaded documents (runtime)
└── chroma_db/                # Persistent vector storage (runtime)
```

## Configuration

All settings can be configured via the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `HF_API_TOKEN` | (empty) | HuggingFace API token (fallback) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `3` | Number of chunks to retrieve |

## Evaluation Metrics

The evaluation dashboard tracks:

| Metric | Description |
|---|---|
| **Retrieval Relevance** | Cosine similarity between query and retrieved chunks (0-100%) |
| **Response Time** | Total time from question to complete answer |
| **Chunks Used** | Number of document chunks retrieved for each answer |
| **Answer Length** | Word count of generated response |
| **Per-Chunk Scores** | Individual relevance score for each retrieved chunk |

## Technology Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Ollama (local) / HuggingFace (cloud) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB |
| Web UI | Streamlit |
| Document Parsing | PyPDF, BeautifulSoup4 |
| Testing | pytest |
| Deployment | Docker, docker-compose |
| Evaluation | Cosine similarity, numpy |
