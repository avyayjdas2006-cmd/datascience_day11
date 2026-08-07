"""
PDF Question Answering Application (RAG)
-----------------------------------------
Streamlit UI wrapping the RAG pipeline in rag_pipeline.py.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from rag_pipeline import (
    load_and_split_pdf,
    build_vector_store,
    build_llm,
    answer_question,
    ConversationMemory,
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")

st.title("📄 PDF RAG Chatbot")
st.markdown("Upload a PDF and ask questions about its content.")

# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of dicts: {role, content, sources}
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

# ---------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None
    
if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Server misconfiguration: GROQ_API_KEY is not set.")
    st.stop()

model_name = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------
# PDF Uploader (Main column, no sidebar)
# ---------------------------------------------------------------------
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.indexed_filename:
    with st.spinner("Processing PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            st.session_state.vector_store = build_vector_store(chunks)
            st.session_state.indexed_filename = uploaded_pdf.name
            st.session_state.memory.clear()
            st.session_state.chat_display = []

            os.unlink(tmp_path)
            st.success("✅ PDF processed successfully!")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")

if st.session_state.indexed_filename:
    st.caption(f"Currently loaded: **{st.session_state.indexed_filename}**")
    if st.button("🗑️ Clear document", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.indexed_filename = None
        st.session_state.memory.clear()
        st.session_state.chat_display = []
        st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
if st.session_state.vector_store:
    for turn in st.session_state.chat_display:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"📎 Sources ({len(turn['sources'])} chunks used)"):
                    for doc in turn["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}**\n\n{doc.page_content}")

    question = st.chat_input("Ask a question about the PDF")

    if question:
        st.session_state.chat_display.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = build_llm(api_key, model_name)
                    answer, sources = answer_question(
                        llm, st.session_state.vector_store, question, st.session_state.memory
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)} chunks used)"):
                            for doc in sources:
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Page {page}**\n\n{doc.page_content}")
                    st.session_state.chat_display.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Something went wrong while calling the Groq API: {e}")
                    st.info(
                        "Common causes: hitting the free-tier rate limit "
                        "(wait a minute and retry), or a network issue."
                    )
