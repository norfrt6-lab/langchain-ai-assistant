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

# --- Page Config ---
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- Sidebar ---
with st.sidebar:
    st.title("RAG AI Assistant")
    st.markdown("---")

    # LLM Status
    st.subheader("LLM Status")
    try:
        _, provider = get_llm()
        if provider == "ollama":
            st.success("Ollama connected")
        else:
            st.info("HuggingFace API connected")
    except ConnectionError as e:
        st.error(str(e))
        if st.button("Retry Connection"):
            reset_llm()
            st.rerun()

    st.markdown("---")

    # File Upload
    st.subheader("Upload Documents")
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
                        st.success(f"{uploaded_file.name}: {num_chunks} chunks added")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

    st.markdown("---")

    # Web URL
    st.subheader("Load Web Page")
    url = st.text_input("Enter URL", placeholder="https://example.com")
    if st.button("Load URL") and url:
        with st.spinner("Loading web page..."):
            try:
                docs = load_web(url)
                chunks = split_documents(docs)
                num_chunks = add_documents(chunks)
                reset_chain()
                st.success(f"Web page loaded: {num_chunks} chunks added")
            except Exception as e:
                st.error(f"Error loading URL: {e}")

    st.markdown("---")

    # Document Info
    st.subheader("Loaded Documents")
    doc_count = get_document_count()
    st.metric("Total Chunks", doc_count)

    sources = list_sources()
    if sources:
        for source in sources:
            st.text(f"- {source}")
    else:
        st.caption("No documents loaded yet.")

    st.markdown("---")

    # Clear All
    if st.button("Clear All Documents", type="secondary"):
        clear_store()
        reset_chain()
        st.session_state.processed_files.clear()
        st.session_state.chat_history.clear()
        st.success("All documents cleared!")
        st.rerun()

# --- Main Chat Area ---
st.header("Chat with Your Documents")

if get_document_count() == 0:
    st.info(
        "Upload documents (PDF, TXT) or load a web page from the sidebar to get started. "
        "Once documents are loaded, you can ask questions about them here."
    )

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, src in enumerate(message["sources"], 1):
                    source_label = src["name"]
                    if "page" in src:
                        source_label += f" (Page {src['page'] + 1})"
                    st.markdown(f"**Source {i}:** {source_label} [{src['type']}]")
                    st.caption(src["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
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
                response_placeholder = st.empty()

                for chunk_text, chunk_sources in ask_question_stream(
                    prompt, st.session_state.chat_history
                ):
                    full_answer += chunk_text
                    sources = chunk_sources
                    response_placeholder.markdown(full_answer + "▌")

                response_placeholder.markdown(full_answer)

                if sources:
                    with st.expander("View Sources"):
                        for i, src in enumerate(sources, 1):
                            source_label = src["name"]
                            if "page" in src:
                                source_label += f" (Page {src['page'] + 1})"
                            st.markdown(
                                f"**Source {i}:** {source_label} [{src['type']}]"
                            )
                            st.caption(src["content"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                })
            except ConnectionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error generating response: {e}")
