import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import src.conversation_store as cs


def _reset_store(tmp_path):
    """Reset the store to use a temporary database."""
    cs.close()
    cs._conn = None
    return Path(tmp_path) / "conversations.db"


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_create_and_list_conversations(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid = cs.create_conversation("Test Chat")
        conversations = cs.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["title"] == "Test Chat"
        assert conversations[0]["id"] == cid
    cs.close()


def test_add_and_get_messages(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid = cs.create_conversation()
        cs.add_message(cid, "user", "Hello")
        cs.add_message(cid, "assistant", "Hi there!", sources=[{"name": "doc.pdf"}])

        messages = cs.get_messages(cid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["sources"] == [{"name": "doc.pdf"}]
    cs.close()


def test_delete_conversation(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid = cs.create_conversation("To Delete")
        cs.add_message(cid, "user", "message")
        cs.delete_conversation(cid)

        conversations = cs.list_conversations()
        assert len(conversations) == 0
        messages = cs.get_messages(cid)
        assert len(messages) == 0
    cs.close()


def test_rename_conversation(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid = cs.create_conversation("Old Name")
        cs.rename_conversation(cid, "New Name")

        conversations = cs.list_conversations()
        assert conversations[0]["title"] == "New Name"
    cs.close()


def test_message_count_in_list(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid = cs.create_conversation()
        cs.add_message(cid, "user", "Q1")
        cs.add_message(cid, "assistant", "A1")
        cs.add_message(cid, "user", "Q2")

        conversations = cs.list_conversations()
        assert conversations[0]["message_count"] == 3
    cs.close()


def test_multiple_conversations_ordered_by_recent(tmp_dir):
    db_path = _reset_store(tmp_dir)
    with patch.object(cs, "DB_PATH", db_path):
        cid1 = cs.create_conversation("First")
        cid2 = cs.create_conversation("Second")
        # Update first conversation to make it more recent
        cs.add_message(cid1, "user", "new message")

        conversations = cs.list_conversations()
        assert conversations[0]["id"] == cid1  # Most recently updated
        assert conversations[1]["id"] == cid2
    cs.close()
