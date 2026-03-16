"""Streamlit frontend for the RAG Document Q&A system."""

import sys
from pathlib import Path

import streamlit as st
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Page config ───
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Document Q&A System")
st.caption("Upload documents and ask questions — powered by RAG with hybrid search")

API_URL = "http://localhost:8000"

# ─── Sidebar: Upload ───
with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF or text file",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "character", "token"],
            index=0,
        )
    with col2:
        chunk_size = st.slider("Chunk Size", 128, 2048, 512, step=64)

    if uploaded_file and st.button("📤 Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                params = {"chunk_strategy": chunk_strategy, "chunk_size": chunk_size}
                response = requests.post(f"{API_URL}/upload", files=files, params=params)

                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"✅ **{result['filename']}** processed into "
                        f"**{result['num_chunks']}** chunks"
                    )
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("⚠️ Cannot connect to the API. Make sure the backend is running.")

    st.divider()

    # Search settings
    st.header("⚙️ Search Settings")
    search_type = st.radio("Search Type", ["hybrid", "semantic"], index=0)
    top_k = st.slider("Number of sources", 1, 10, 5)

    if st.button("🗑️ Clear All Documents"):
        try:
            requests.post(f"{API_URL}/clear")
            st.success("Vector store cleared!")
        except requests.ConnectionError:
            st.error("Cannot connect to the API.")

# ─── Main: Q&A ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Source {i}:** {src['source']} (Page {src['page']}) "
                        f"— Relevance: {src['relevance_score']:.2%}"
                    )
                    st.caption(src["chunk_preview"])

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "search_type": search_type,
                        "top_k": top_k,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result["answer"])

                    with st.expander("📚 View Sources"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(
                                f"**Source {i}:** {src['source']} (Page {src['page']}) "
                                f"— Relevance: {src['relevance_score']:.2%}"
                            )
                            st.caption(src["chunk_preview"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)

            except requests.ConnectionError:
                st.error(
                    "⚠️ Cannot connect to the API. Start the backend with:\n"
                    "`uvicorn src.api.main:app --reload`"
                )