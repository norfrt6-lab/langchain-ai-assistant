import time
import numpy as np
from src.embeddings import get_embeddings


def calculate_retrieval_relevance(query, documents):
    """Calculate cosine similarity between query and each retrieved document."""
    if not documents:
        return {"scores": [], "avg_score": 0.0}

    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)

    scores = []
    for doc in documents:
        doc_embedding = embeddings.embed_query(doc.page_content[:500])
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        scores.append(round(float(similarity), 4))

    return {
        "scores": scores,
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def calculate_response_metrics(response_time, sources, answer):
    """Calculate response quality metrics."""
    return {
        "response_time": round(response_time, 2),
        "chunks_used": len(sources),
        "answer_length": len(answer),
        "answer_words": len(answer.split()),
    }


def evaluate_response(query, answer, context_docs, sources, response_time):
    """Run full evaluation on a RAG response."""
    relevance = calculate_retrieval_relevance(query, context_docs)
    response_metrics = calculate_response_metrics(response_time, sources, answer)

    return {
        "relevance": relevance,
        "response_time": response_metrics["response_time"],
        "chunks_used": response_metrics["chunks_used"],
        "answer_length": response_metrics["answer_length"],
        "answer_words": response_metrics["answer_words"],
    }
