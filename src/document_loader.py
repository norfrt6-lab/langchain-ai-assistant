import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from src.config import DATA_DIR


def load_pdf(uploaded_file):
    """Load a PDF file and return documents with metadata."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["filename"] = uploaded_file.name

    return documents


def load_txt(uploaded_file):
    """Load a TXT file and return documents with metadata."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "txt"
        doc.metadata["filename"] = uploaded_file.name

    return documents


def load_web(url):
    """Load a web page and return documents with metadata."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "web"
        doc.metadata["url"] = url

    return documents
