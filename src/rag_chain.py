from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from src.llm import get_llm
from src.vector_store import get_retriever

SYSTEM_PROMPT = """Answer based on the context below. Be concise. If the context lacks the answer, say so.

Context:
{context}

Chat history:
{chat_history}"""

_rag_chain = None


def _format_chat_history(history, max_turns=3):
    """Format recent chat history as a string for the prompt."""
    if not history:
        return "No previous conversation."

    recent = history[-max_turns:]
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def get_rag_chain():
    """Create the RAG chain."""
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    llm, _ = get_llm()
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    _rag_chain = create_retrieval_chain(retriever, document_chain)
    return _rag_chain


def _extract_sources(docs):
    """Extract source information from retrieved documents."""
    sources = []
    for doc in docs:
        meta = doc.metadata
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "type": meta.get("source_type", "unknown"),
        }
        if "filename" in meta:
            source_info["name"] = meta["filename"]
        elif "url" in meta:
            source_info["name"] = meta["url"]
        else:
            source_info["name"] = meta.get("source", "Unknown")
        if "page" in meta:
            source_info["page"] = meta["page"]
        sources.append(source_info)
    return sources


def ask_question(question, chat_history=None):
    """Ask a question and get an answer with sources."""
    chain = get_rag_chain()
    formatted_history = _format_chat_history(chat_history or [])

    result = chain.invoke({
        "input": question,
        "chat_history": formatted_history,
    })

    return {
        "answer": result["answer"],
        "sources": _extract_sources(result.get("context", [])),
    }


def ask_question_stream(question, chat_history=None):
    """Ask a question with streaming response. Yields (chunk, sources) tuples."""
    chain = get_rag_chain()
    formatted_history = _format_chat_history(chat_history or [])

    sources = []
    for chunk in chain.stream({
        "input": question,
        "chat_history": formatted_history,
    }):
        if "context" in chunk:
            sources = _extract_sources(chunk["context"])
        if "answer" in chunk:
            yield chunk["answer"], sources


def reset_chain():
    """Reset the RAG chain (e.g., after clearing documents)."""
    global _rag_chain
    _rag_chain = None
