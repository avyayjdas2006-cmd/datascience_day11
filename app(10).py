import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ==========================
# Load API Key
# ==========================

load_dotenv()

load_dotenv()

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

# Make API key available to rag.py
os.environ["GOOGLE_API_KEY"] = api_key

from rag import process_pdf, ask_pdf

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="PDF Question Answering Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Question Answering Assistant")
st.write("Upload a PDF and ask questions about its contents using **Google Gemini + LangChain + ChromaDB**.")

# ==========================
# Session State
# ==========================

if "history" not in st.session_state:
    st.session_state.history = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        if st.button("📄 Index PDF", use_container_width=True):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getbuffer())
                pdf_path = tmp.name

            with st.spinner("Indexing PDF..."):

                try:
                    result = process_pdf(pdf_path)

                    st.success(result)

                    st.session_state.indexed = True
                    st.session_state.history = []

                except Exception as e:
                    st.error(str(e))

                finally:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

# ==========================
# Chat History
# ==========================

for question, answer in st.session_state.history:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(answer)

# ==========================
# Chat Input
# ==========================

if st.session_state.indexed:

    question = st.chat_input("Ask a question about your PDF...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            answer, st.session_state.history = ask_pdf(
                question,
                st.session_state.history,
            )

        with st.chat_message("assistant"):
            st.write(answer)

else:

    st.info("📄 Please upload and index a PDF first.")