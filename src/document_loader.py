import os
import csv
import io
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def _save_uploaded_file(uploaded_file):
    """Save an uploaded file to the data directory and return the path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf(uploaded_file):
    """Load a PDF file and return documents with metadata."""
    try:
        file_path = _save_uploaded_file(uploaded_file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content found in PDF '{uploaded_file.name}'")

        for doc in documents:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["filename"] = uploaded_file.name

        logger.info(f"Loaded PDF '{uploaded_file.name}': {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF '{uploaded_file.name}': {e}")
        raise


def load_txt(uploaded_file):
    """Load a TXT file and return documents with metadata."""
    try:
        file_path = _save_uploaded_file(uploaded_file)
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_type"] = "txt"
            doc.metadata["filename"] = uploaded_file.name

        logger.info(f"Loaded TXT '{uploaded_file.name}': {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to load TXT '{uploaded_file.name}': {e}")
        raise


def load_docx(uploaded_file):
    """Load a DOCX file and return documents with metadata."""
    try:
        file_path = _save_uploaded_file(uploaded_file)
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_type"] = "docx"
            doc.metadata["filename"] = uploaded_file.name

        logger.info(f"Loaded DOCX '{uploaded_file.name}': {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to load DOCX '{uploaded_file.name}': {e}")
        raise


def load_csv(uploaded_file):
    """Load a CSV file and return documents with metadata."""
    try:
        content = uploaded_file.getbuffer().tobytes().decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file '{uploaded_file.name}' is empty")

        header = rows[0]
        text_parts = []
        for row in rows[1:]:
            row_text = ", ".join(f"{h}: {v}" for h, v in zip(header, row) if v.strip())
            if row_text:
                text_parts.append(row_text)

        if not text_parts:
            raise ValueError(f"CSV file '{uploaded_file.name}' has no data rows")

        full_text = "\n".join(text_parts)
        documents = [
            Document(
                page_content=full_text,
                metadata={
                    "source_type": "csv",
                    "filename": uploaded_file.name,
                    "rows": len(rows) - 1,
                    "columns": len(header),
                },
            )
        ]
        logger.info(f"Loaded CSV '{uploaded_file.name}': {len(rows)-1} rows, {len(header)} columns")
        return documents
    except Exception as e:
        logger.error(f"Failed to load CSV '{uploaded_file.name}': {e}")
        raise


def load_web(url):
    """Load a web page and return documents with metadata."""
    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: '{url}'. Must start with http:// or https://")

        loader = WebBaseLoader(url)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content found at '{url}'")

        for doc in documents:
            doc.metadata["source_type"] = "web"
            doc.metadata["url"] = url

        logger.info(f"Loaded web page '{url}': {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Failed to load web page '{url}': {e}")
        raise
