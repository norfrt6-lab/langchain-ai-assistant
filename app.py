import time
import streamlit as st
from src.document_loader import load_pdf, load_txt, load_web
from src.text_splitter import split_documents
from src.vector_store import (
    add_documents,
    get_document_count,
    list_sources,
    clear_store,
)
from src.llm import get_llm, reset_llm
from src.rag_chain import ask_question_stream, reset_chain
from src.evaluation import evaluate_response
from src.styles import CUSTOM_CSS, get_source_card_html, get_metrics_html

# --- Page Config ---
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Inject Custom CSS ---
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        '<div class="main-header"><h1>RAG AI Assistant</h1>'
        "<p>Chat with your documents</p></div>",
        unsafe_allow_html=True,
    )

    # LLM Status
    st.markdown("### ⚡ LLM Status")
    try:
        _, provider = get_llm()
        if provider == "ollama":
            st.markdown(
                '<span class="status-badge status-connected">Ollama Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-connected">HuggingFace Connected</span>',
                unsafe_allow_html=True,
            )
    except ConnectionError as e:
        st.markdown(
            '<span class="status-badge status-disconnected">Disconnected</span>',
            unsafe_allow_html=True,
        )
        st.caption(str(e))
        if st.button("🔄 Retry Connection"):
            reset_llm()
            st.rerun()

    st.markdown("---")

    # File Upload
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.processed_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        if uploaded_file.name.endswith(".pdf"):
                            docs = load_pdf(uploaded_file)
                        else:
                            docs = load_txt(uploaded_file)

                        chunks = split_documents(docs)
                        num_chunks = add_documents(chunks)
                        st.session_state.processed_files.add(file_id)
                        reset_chain()
                        st.success(f"✅ {uploaded_file.name}: {num_chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")

    st.markdown("---")

    # Web URL
    st.markdown("### 🌐 Load Web Page")
    url = st.text_input("Enter URL", placeholder="https://example.com")
    if st.button("Load URL") and url:
        with st.spinner("Loading web page..."):
            try:
                docs = load_web(url)
                chunks = split_documents(docs)
                num_chunks = add_documents(chunks)
                reset_chain()
                st.success(f"✅ Loaded: {num_chunks} chunks")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")

    # Document Info
    st.markdown("### 📊 Loaded Documents")
    doc_count = get_document_count()
    st.metric("Total Chunks", doc_count)

    sources = list_sources()
    if sources:
        for source in sources:
            st.text(f"  • {source}")
    else:
        st.caption("No documents loaded yet.")

    st.markdown("---")

    # Clear All
    if st.button("🗑️ Clear All Documents", type="secondary"):
        clear_store()
        reset_chain()
        st.session_state.processed_files.clear()
        st.session_state.chat_history.clear()
        st.session_state.eval_results.clear()
        st.success("All cleared!")
        st.rerun()

# --- Main Area with Tabs ---
tab_chat, tab_eval = st.tabs(["💬 Chat", "📈 Evaluation"])

# ==================== CHAT TAB ====================
with tab_chat:
    st.markdown(
        '<div class="main-header"><h1>💬 Chat with Your Documents</h1>'
        "<p>Upload documents and ask questions to get AI-powered answers</p></div>",
        unsafe_allow_html=True,
    )

    if get_document_count() == 0:
        st.info(
            "👈 Upload documents (PDF, TXT) or load a web page from the sidebar to get started."
        )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "metrics" in message and message["metrics"]:
                st.markdown(
                    get_metrics_html(message["metrics"]),
                    unsafe_allow_html=True,
                )

            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(
                            get_source_card_html(src, i),
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if get_document_count() == 0:
                response_text = "Please upload some documents first before asking questions."
                st.markdown(response_text)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
            else:
                try:
                    full_answer = ""
                    sources = []
                    context_docs = []
                    response_placeholder = st.empty()
                    start_time = time.time()

                    for chunk_text, chunk_sources, chunk_context in ask_question_stream(
                        prompt, st.session_state.chat_history
                    ):
                        full_answer += chunk_text
                        sources = chunk_sources
                        context_docs = chunk_context
                        response_placeholder.markdown(full_answer + "▌")

                    response_time = time.time() - start_time
                    response_placeholder.markdown(full_answer)

                    # Evaluate response
                    metrics = evaluate_response(
                        prompt, full_answer, context_docs, sources, response_time
                    )

                    # Display metrics
                    st.markdown(
                        get_metrics_html(metrics), unsafe_allow_html=True
                    )

                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(
                                    get_source_card_html(src, i),
                                    unsafe_allow_html=True,
                                )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources,
                        "metrics": metrics,
                    })

                    # Store for evaluation tab
                    st.session_state.eval_results.append({
                        "question": prompt,
                        "answer": full_answer[:100] + "..." if len(full_answer) > 100 else full_answer,
                        **metrics,
                    })

                except ConnectionError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error generating response: {e}")

# ==================== EVALUATION TAB ====================
with tab_eval:
    st.markdown(
        '<div class="main-header"><h1>📈 RAG Evaluation Dashboard</h1>'
        "<p>Monitor retrieval quality and response performance</p></div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.eval_results:
        st.info("No evaluation data yet. Ask some questions in the Chat tab first.")
    else:
        # Summary metrics
        results = st.session_state.eval_results
        avg_relevance = sum(
            r.get("relevance", {}).get("avg_score", 0) for r in results
        ) / len(results)
        avg_time = sum(r.get("response_time", 0) for r in results) / len(results)
        avg_chunks = sum(r.get("chunks_used", 0) for r in results) / len(results)
        total_queries = len(results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Avg Relevance", f"{avg_relevance:.1%}")
        with col3:
            st.metric("Avg Response Time", f"{avg_time:.1f}s")
        with col4:
            st.metric("Avg Chunks Used", f"{avg_chunks:.1f}")

        st.markdown("---")

        # Detailed results table
        st.subheader("Query History")
        for i, result in enumerate(reversed(results), 1):
            relevance = result.get("relevance", {}).get("avg_score", 0)
            if relevance >= 0.7:
                relevance_icon = "🟢"
            elif relevance >= 0.4:
                relevance_icon = "🟡"
            else:
                relevance_icon = "🔴"

            with st.expander(
                f"Q{len(results) - i + 1}: {result['question'][:80]}... "
                f"| {relevance_icon} {relevance:.1%} | ⏱️ {result.get('response_time', 0):.1f}s"
            ):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Relevance", f"{relevance:.1%}")
                with col_b:
                    st.metric("Response Time", f"{result.get('response_time', 0):.2f}s")
                with col_c:
                    st.metric("Chunks Used", result.get("chunks_used", 0))

                st.markdown("**Answer Preview:**")
                st.caption(result.get("answer", ""))

                # Per-chunk relevance scores
                chunk_scores = result.get("relevance", {}).get("scores", [])
                if chunk_scores:
                    st.markdown("**Per-Chunk Relevance:**")
                    for j, score in enumerate(chunk_scores, 1):
                        bar_pct = int(score * 100)
                        st.progress(score, text=f"Chunk {j}: {score:.1%}")

# --- Footer ---
st.markdown(
    '<div class="app-footer">'
    "RAG AI Assistant — Senior Project 2026<br>"
    "Built with LangChain • ChromaDB • Ollama • Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
