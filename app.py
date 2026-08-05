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
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")

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
# Sidebar: configuration
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    api_key = st.text_input(
        "Groq API Key",
        value=api_key or "",
        type="password",
        help="Free key: https://console.groq.com/keys",
    )

    model_name = st.selectbox(
        "Chat model",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
        help="llama-3.1-8b-instant is fast and free-tier friendly.",
    )

    st.divider()
    st.subheader("📄 Upload a PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.indexed_filename:
        if not api_key:
            st.warning("Enter your Groq API key above before indexing a PDF.")
        else:
            with st.spinner("Reading PDF, splitting into chunks, and building the search index..."):
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
                    st.success(f"Indexed **{uploaded_pdf.name}** — {len(chunks)} chunks ready for Q&A.")
                except Exception as e:
                    st.error(f"Failed to index PDF: {e}")

    if st.session_state.indexed_filename:
        st.caption(f"✅ Currently indexed: **{st.session_state.indexed_filename}**")
        if st.button("🗑️ Clear document & conversation", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.indexed_filename = None
            st.session_state.memory.clear()
            st.session_state.chat_display = []
            st.rerun()

    st.divider()
    st.caption(
        "Your questions and retrieved PDF excerpts are sent to Groq's API to generate "
        "answers and are not stored by this app. Avoid uploading documents with sensitive "
        "data you don't want processed by a third-party API. Embeddings are generated "
        "locally and never leave your machine."
    )

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
if not st.session_state.vector_store:
    st.info("👈 Upload a PDF in the sidebar and enter your Groq API key to get started.")
else:
    for turn in st.session_state.chat_display:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"📎 Sources ({len(turn['sources'])} chunks used)"):
                    for doc in turn["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}**\n\n{doc.page_content}")

    question = st.chat_input("Ask a question about the PDF...")

    if question:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            st.session_state.chat_display.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {model_name}..."):
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
                            "Common causes: an invalid API key, hitting the free-tier rate limit "
                            "(wait a minute and retry), or a network issue."
                        )

st.divider()
st.caption(
    "RAG pipeline: PyPDFLoader → RecursiveCharacterTextSplitter → local Sentence-Transformer "
    "embeddings → FAISS → Groq (Llama 3.1) chat. Answers are AI-generated and grounded in the "
    "uploaded document — always verify against the original source for important decisions."
)
