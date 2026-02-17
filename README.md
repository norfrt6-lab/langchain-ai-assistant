# RAG AI Assistant

A Retrieval-Augmented Generation (RAG) AI Assistant built with LangChain, ChromaDB, and Streamlit. Upload your documents (PDF, TXT, or web pages) and chat with them using a local, free LLM.

## Features

- **Multi-format document support**: PDF, TXT files, and web pages
- **Local & free**: Uses Ollama for local LLM inference (no API costs)
- **Persistent storage**: ChromaDB stores document embeddings across sessions
- **Chat interface**: Streamlit-based chat UI with conversation history
- **Source attribution**: See which document chunks were used to generate each answer
- **Auto-fallback**: Falls back to HuggingFace API if Ollama is unavailable

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
LLM (Ollama / HuggingFace) --> Answer with Sources
```

## Prerequisites

- **Python 3.10+**
- **Ollama** (recommended) - for free local LLM

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
# Edit .env if needed (defaults work for Ollama)
```

## Usage

1. Make sure Ollama is running (if using Ollama)
2. Start the application:

```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`
4. Upload documents from the sidebar (PDF or TXT) or enter a web URL
5. Start chatting with your documents!

## Project Structure

```
langchain-ai-assistant/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
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
│   └── rag_chain.py          # RAG pipeline chain
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
| `TOP_K_RESULTS` | `4` | Number of chunks to retrieve |

## Technology Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| LLM | Ollama (local) / HuggingFace (cloud) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB |
| Web UI | Streamlit |
| Document Parsing | PyPDF, BeautifulSoup4 |
