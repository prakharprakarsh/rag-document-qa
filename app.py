"""Combined app for HuggingFace Spaces deployment.

This runs the RAG pipeline directly (no separate FastAPI server)
so it works within HuggingFace Spaces' single-process constraint.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents, compare_strategies
from src.retrieval.vector_store import add_documents, clear_vector_store, similarity_search
from src.generation.llm_chain import ask_question, format_context, get_llm

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")
st.title("📄 RAG Document Q&A System")
st.caption("Upload documents and ask natural language questions — powered by RAG with hybrid search")

# ─── Sidebar ───
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    col1, col2 = st.columns(2)
    with col1:
        chunk_strategy = st.selectbox("Chunking", ["recursive", "character", "token"])
    with col2:
        chunk_size = st.slider("Chunk Size", 128, 2048, 512, step=64)

    if uploaded_file and st.button("📤 Process", type="primary"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            documents = load_document(tmp_path)
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
            chunks = chunk_documents(documents, strategy=chunk_strategy, chunk_size=chunk_size)
            add_documents(chunks)
            st.success(f"✅ {uploaded_file.name} → {len(chunks)} chunks")
            Path(tmp_path).unlink(missing_ok=True)

    st.divider()
    search_type = st.radio("Search", ["semantic", "hybrid"], index=0)
    top_k = st.slider("Sources", 1, 10, 5)

    if st.button("🗑️ Clear"):
        clear_vector_store()
        st.success("Cleared!")

# ─── Chat ───
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(question, search_type=search_type, k=top_k)
            st.markdown(result["answer"])

            with st.expander("📚 Sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**{i}.** {src['source']} (p.{src['page']}) — {src['relevance_score']:.2%}")
                    st.caption(src["chunk_preview"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
            })