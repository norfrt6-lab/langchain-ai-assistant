"""FastAPI REST API for the RAG AI Assistant."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import conversation_store as cs
from src.document_loader import load_csv, load_docx, load_pdf, load_txt, load_web
from src.evaluation import evaluate_response
from src.llm import get_llm, reset_llm
from src.rag_chain import ask_question, reset_chain
from src.text_splitter import split_documents
from src.vector_store import add_documents, clear_store, get_document_count, list_sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    cs.close()


app = FastAPI(
    title="RAG AI Assistant API",
    description="REST API for document-based question answering with RAG",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: int | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    evaluation: dict | None = None
    conversation_id: int


class URLRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://")


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# --- Health ---


@app.get("/health")
def health():
    try:
        _, provider = get_llm()
        llm_status = provider
    except ConnectionError:
        llm_status = "disconnected"

    return {
        "status": "ok",
        "llm": llm_status,
        "documents": get_document_count(),
        "sources": list_sources(),
    }


# --- Documents ---


@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    loaders = {"pdf": load_pdf, "txt": load_txt, "docx": load_docx, "csv": load_csv}
    loader = loaders.get(ext)
    if not loader:
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    documents = loader(file)
    chunks = split_documents(documents)
    count = add_documents(chunks)
    return {"filename": file.filename, "chunks": count}


@app.post("/documents/url")
def load_url(req: URLRequest):
    documents = load_web(req.url)
    chunks = split_documents(documents)
    count = add_documents(chunks)
    return {"url": req.url, "chunks": count}


@app.get("/documents")
def list_documents():
    return {"count": get_document_count(), "sources": list_sources()}


@app.delete("/documents")
def clear_documents():
    clear_store()
    reset_chain()
    return {"status": "cleared"}


# --- Questions ---


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if get_document_count() == 0:
        raise HTTPException(400, "No documents loaded. Upload documents first.")

    cid = req.conversation_id
    if cid is None:
        cid = cs.create_conversation(req.question[:50])

    history = cs.get_messages(cid)
    cs.add_message(cid, "user", req.question)

    start = time.time()
    result = ask_question(req.question, history)
    elapsed = time.time() - start

    cs.add_message(cid, "assistant", result["answer"], sources=result["sources"])

    evaluation = evaluate_response(
        req.question,
        result["answer"],
        [],
        result["sources"],
        elapsed,
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        evaluation=evaluation,
        conversation_id=cid,
    )


# --- Conversations ---


@app.get("/conversations", response_model=list[ConversationResponse])
def list_conversations():
    return cs.list_conversations()


@app.get("/conversations/{cid}/messages")
def get_messages(cid: int):
    messages = cs.get_messages(cid)
    return {"conversation_id": cid, "messages": messages}


@app.patch("/conversations/{cid}")
def rename_conversation(cid: int, req: RenameRequest):
    cs.rename_conversation(cid, req.title)
    return {"status": "renamed"}


@app.delete("/conversations/{cid}")
def delete_conversation(cid: int):
    cs.delete_conversation(cid)
    return {"status": "deleted"}


# --- LLM ---


@app.post("/llm/reconnect")
def reconnect_llm():
    reset_llm()
    reset_chain()
    try:
        _, provider = get_llm()
        return {"status": "connected", "provider": provider}
    except ConnectionError as e:
        raise HTTPException(503, str(e)) from None
