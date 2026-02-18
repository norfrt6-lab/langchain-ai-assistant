from unittest.mock import patch, MagicMock
from src.rag_chain import _format_chat_history, reset_chain
import src.rag_chain as rag_module


def test_format_chat_history_empty():
    """Test formatting empty chat history."""
    result = _format_chat_history([])
    assert result == "No previous conversation."


def test_format_chat_history_none():
    """Test formatting None chat history."""
    result = _format_chat_history(None)
    assert result == "No previous conversation."


def test_format_chat_history_with_messages():
    """Test formatting chat history with messages."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = _format_chat_history(history)

    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result


def test_format_chat_history_max_turns():
    """Test that chat history respects max_turns limit."""
    history = [
        {"role": "user", "content": f"Message {i}"}
        for i in range(10)
    ]

    result = _format_chat_history(history, max_turns=2)

    assert "Message 8" in result
    assert "Message 9" in result
    assert "Message 0" not in result


def test_reset_chain():
    """Test that reset_chain clears the cached chain."""
    rag_module._rag_chain = MagicMock()

    reset_chain()

    assert rag_module._rag_chain is None


def test_ask_question_with_mocked_chain():
    """Test ask_question with a fully mocked chain."""
    rag_module._rag_chain = None

    mock_chain = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Test content from document"
    mock_doc.metadata = {
        "source_type": "pdf",
        "filename": "test.pdf",
        "page": 0,
        "chunk_index": 0,
    }
    mock_chain.invoke.return_value = {
        "answer": "Test answer",
        "context": [mock_doc],
    }

    with patch("src.rag_chain.get_rag_chain", return_value=mock_chain):
        from src.rag_chain import ask_question

        result = ask_question("What is this?")

        assert result["answer"] == "Test answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["name"] == "test.pdf"
        assert result["sources"][0]["type"] == "pdf"
