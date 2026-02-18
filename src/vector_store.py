import logging
from langchain_chroma import Chroma
from src.config import CHROMA_DB_DIR, TOP_K_RESULTS
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vector_store = None


def get_vector_store():
    """Get or create the ChromaDB vector store (singleton)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name="documents",
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DB_DIR,
        )
    return _vector_store


def add_documents(chunks):
    """Add document chunks to the vector store."""
    store = get_vector_store()
    store.add_documents(chunks)
    logger.info(f"Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search(query, k=None):
    """Search for similar documents."""
    if k is None:
        k = TOP_K_RESULTS
    store = get_vector_store()
    return store.similarity_search(query, k=k)


def get_retriever(k=None):
    """Get a retriever for the RAG chain."""
    if k is None:
        k = TOP_K_RESULTS
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k})


def get_document_count():
    """Get the total number of chunks in the store."""
    store = get_vector_store()
    return store._collection.count()


def list_sources():
    """List all unique document sources."""
    store = get_vector_store()
    results = store._collection.get(include=["metadatas"])
    sources = set()
    for meta in results.get("metadatas", []):
        if "filename" in meta:
            sources.add(meta["filename"])
        elif "url" in meta:
            sources.add(meta["url"])
    return sorted(sources)


def clear_store():
    """Clear all documents from the vector store."""
    global _vector_store
    store = get_vector_store()
    store._collection.delete(where={"chunk_index": {"$gte": 0}})
    _vector_store = None
    logger.info("Cleared all documents from vector store")
