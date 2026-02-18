from langchain_core.documents import Document
from src.text_splitter import split_documents


def test_split_documents_basic():
    """Test that documents are split into chunks."""
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    chunks = split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) > 0


def test_split_documents_preserves_metadata():
    """Test that original metadata is preserved in chunks."""
    docs = [
        Document(
            page_content="Word " * 500,
            metadata={"source": "test.pdf", "source_type": "pdf"},
        )
    ]

    chunks = split_documents(docs)

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["source_type"] == "pdf"


def test_split_documents_adds_chunk_index():
    """Test that chunk_index metadata is added."""
    docs = [Document(page_content="Hello world. " * 300, metadata={})]

    chunks = split_documents(docs)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_split_short_document():
    """Test that a short document stays as one chunk."""
    docs = [Document(page_content="Short text.", metadata={})]

    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_split_empty_list():
    """Test that an empty list returns empty."""
    chunks = split_documents([])
    assert chunks == []
