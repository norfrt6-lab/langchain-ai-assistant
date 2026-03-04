import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.conversation_store as cs

# Patch LLM before importing api module
with patch("src.llm.get_llm", return_value=(MagicMock(), "mock")):
    from api import app


@pytest.fixture(autouse=True)
def tmp_db():
    """Use temporary database for each test."""
    cs.close()
    cs._conn = None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db_path = Path(d) / "test.db"
        with patch.object(cs, "DB_PATH", db_path):
            yield
            cs.close()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    with (
        patch("api.get_llm", return_value=(MagicMock(), "ollama")),
        patch("api.get_document_count", return_value=5),
        patch("api.list_sources", return_value=["doc.pdf"]),
    ):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["llm"] == "ollama"
        assert data["documents"] == 5


def test_list_documents(client):
    with (
        patch("api.get_document_count", return_value=10),
        patch("api.list_sources", return_value=["a.pdf", "b.txt"]),
    ):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json()["count"] == 10


def test_clear_documents(client):
    with patch("api.clear_store"), patch("api.reset_chain"):
        resp = client.delete("/documents")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"


def test_ask_requires_documents(client):
    with patch("api.get_document_count", return_value=0):
        resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 400


def test_list_conversations(client):
    cs.create_conversation("Test")
    resp = client.get("/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["title"] == "Test"


def test_delete_conversation(client):
    cid = cs.create_conversation("To Delete")
    resp = client.delete(f"/conversations/{cid}")
    assert resp.status_code == 200
    assert len(cs.list_conversations()) == 0


def test_rename_conversation(client):
    cid = cs.create_conversation("Old")
    resp = client.patch(f"/conversations/{cid}", json={"title": "New"})
    assert resp.status_code == 200
    convos = cs.list_conversations()
    assert convos[0]["title"] == "New"


def test_get_messages(client):
    cid = cs.create_conversation("Chat")
    cs.add_message(cid, "user", "Hello")
    resp = client.get(f"/conversations/{cid}/messages")
    assert resp.status_code == 200
    assert len(resp.json()["messages"]) == 1


def test_reconnect_llm(client):
    with (
        patch("api.reset_llm"),
        patch("api.reset_chain"),
        patch("api.get_llm", return_value=(MagicMock(), "ollama")),
    ):
        resp = client.post("/llm/reconnect")
        assert resp.status_code == 200
        assert resp.json()["provider"] == "ollama"


def test_upload_unsupported_file(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("test.xyz", BytesIO(b"content"), "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_load_url_invalid(client):
    resp = client.post("/documents/url", json={"url": "not-a-url"})
    assert resp.status_code == 422
