import os
import tempfile
from unittest.mock import MagicMock, patch
from src.document_loader import load_pdf, load_txt, load_web


class FakeUploadedFile:
    """Mock Streamlit uploaded file."""

    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


def test_load_txt(tmp_path, monkeypatch):
    """Test loading a TXT file."""
    monkeypatch.setattr("src.document_loader.DATA_DIR", str(tmp_path))

    content = b"Hello, this is a test document with some text content."
    fake_file = FakeUploadedFile("test.txt", content)

    docs = load_txt(fake_file)

    assert len(docs) >= 1
    assert docs[0].metadata["source_type"] == "txt"
    assert docs[0].metadata["filename"] == "test.txt"
    assert "Hello" in docs[0].page_content


def test_load_pdf(tmp_path, monkeypatch):
    """Test loading a PDF file (mocked)."""
    monkeypatch.setattr("src.document_loader.DATA_DIR", str(tmp_path))

    mock_docs = [
        MagicMock(
            page_content="PDF content here",
            metadata={"source": "test.pdf", "page": 0},
        )
    ]

    with patch("src.document_loader.PyPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = mock_docs
        fake_file = FakeUploadedFile("test.pdf", b"%PDF-1.4 fake content")

        docs = load_pdf(fake_file)

        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "pdf"
        assert docs[0].metadata["filename"] == "test.pdf"


def test_load_web():
    """Test loading a web page (mocked)."""
    mock_docs = [
        MagicMock(
            page_content="Web page content",
            metadata={"source": "https://example.com"},
        )
    ]

    with patch("src.document_loader.WebBaseLoader") as mock_loader:
        mock_loader.return_value.load.return_value = mock_docs

        docs = load_web("https://example.com")

        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "web"
        assert docs[0].metadata["url"] == "https://example.com"


def test_load_txt_creates_data_dir(tmp_path, monkeypatch):
    """Test that data directory is created if it doesn't exist."""
    new_dir = str(tmp_path / "new_data")
    monkeypatch.setattr("src.document_loader.DATA_DIR", new_dir)

    fake_file = FakeUploadedFile("test.txt", b"Some content")
    load_txt(fake_file)

    assert os.path.exists(new_dir)
