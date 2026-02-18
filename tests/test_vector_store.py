import tempfile
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import src.vector_store as vs_module


def _make_mock_store():
    """Create a mock vector store with a mock collection."""
    mock_store = MagicMock()
    mock_collection = MagicMock()
    mock_store._collection = mock_collection
    return mock_store, mock_collection


def test_add_documents():
    """Test adding documents to the vector store."""
    mock_store, _ = _make_mock_store()
    vs_module._vector_store = mock_store

    chunks = [
        Document(page_content="chunk 1", metadata={"chunk_index": 0}),
        Document(page_content="chunk 2", metadata={"chunk_index": 1}),
    ]

    count = vs_module.add_documents(chunks)

    assert count == 2
    mock_store.add_documents.assert_called_once_with(chunks)

    vs_module._vector_store = None


def test_search():
    """Test searching for similar documents."""
    mock_store, _ = _make_mock_store()
    mock_store.similarity_search.return_value = [
        Document(page_content="result", metadata={})
    ]
    vs_module._vector_store = mock_store

    results = vs_module.search("test query", k=2)

    assert len(results) == 1
    mock_store.similarity_search.assert_called_once_with("test query", k=2)

    vs_module._vector_store = None


def test_get_document_count():
    """Test getting document count."""
    mock_store, mock_collection = _make_mock_store()
    mock_collection.count.return_value = 42
    vs_module._vector_store = mock_store

    count = vs_module.get_document_count()

    assert count == 42

    vs_module._vector_store = None


def test_list_sources():
    """Test listing unique sources."""
    mock_store, mock_collection = _make_mock_store()
    mock_collection.get.return_value = {
        "metadatas": [
            {"filename": "doc1.pdf"},
            {"filename": "doc1.pdf"},
            {"url": "https://example.com"},
            {"filename": "doc2.txt"},
        ]
    }
    vs_module._vector_store = mock_store

    sources = vs_module.list_sources()

    assert "doc1.pdf" in sources
    assert "doc2.txt" in sources
    assert "https://example.com" in sources
    assert len(sources) == 3

    vs_module._vector_store = None


def test_clear_store():
    """Test clearing the vector store."""
    mock_store, mock_collection = _make_mock_store()
    vs_module._vector_store = mock_store

    vs_module.clear_store()

    mock_collection.delete.assert_called_once()
    assert vs_module._vector_store is None
