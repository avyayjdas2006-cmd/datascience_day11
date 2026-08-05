# 📄 PDF Question Answering App (RAG)

**Participant Name:** Avyay J Das
**MUID:** avyayjdas@mulearn

**Live Deployment:** _<paste your Streamlit Community Cloud URL here after deploying — see "Deploying" section below>_

---

## 📌 Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** application that lets a
user upload any PDF document, ask natural-language questions about its content,
and get accurate, context-grounded answers — including follow-up questions that
depend on earlier turns in the conversation.

Pipeline:

1. **Load** the uploaded PDF with `PyPDFLoader`.
2. **Split** the document into overlapping chunks with
   `RecursiveCharacterTextSplitter` so context isn't lost at chunk boundaries.
3. **Embed** each chunk using the free, local
   `sentence-transformers/all-MiniLM-L6-v2` model.
4. **Store** the embeddings in a **ChromaDB** in-memory vector store for
   similarity search.
5. **Retrieve** the top-k most relevant chunks for a user's query.
6. **Generate** an answer with a free, fast LLM (**Groq's Llama 3.1**) via
   LangChain's `ConversationalRetrievalChain`.
7. **Remember** the conversation using LangChain's `ConversationBufferMemory`,
   so follow-up questions like "what about the second one?" resolve correctly.
8. **Serve** all of this through an interactive **Streamlit** chat UI.

---

## 🤖 Technologies Used

| Purpose            | Technology                                             |
|---------------------|--------------------------------------------------------|
| Orchestration        | LangChain                                              |
| PDF loading           | `PyPDFLoader` (langchain-community)                    |
| Chunking               | `RecursiveCharacterTextSplitter`                        |
| Embeddings              | Sentence-Transformers (`all-MiniLM-L6-v2`)              |
| Vector database           | ChromaDB (in-memory, via `langchain-chroma`)          |
| LLM (free tier)             | Groq API — `llama-3.1-8b-instant`                    |
| Conversation memory           | `ConversationBufferMemory`                          |
| UI                               | Streamlit (`st.chat_message`, `st.chat_input`)      |
| Deployment                        | Streamlit Community Cloud                          |

**Why Groq?** It offers a generous free tier with very low latency, so the
answer-generation step feels near-instant compared to many other free LLM
options — good for a responsive chat experience.

---

## 🧠 Memory Implementation

Conversational memory is implemented using LangChain's
`ConversationBufferMemory`, wired into a `ConversationalRetrievalChain`:

- Every user question and assistant answer is stored in the memory buffer
  (`chat_history`) for the duration of the Streamlit session.
- On each new question, the chain first uses the LLM + chat history to
  **condense the question** into a standalone query (resolving pronouns like
  "it", "that", "the second point" against prior turns).
- The standalone query is then used to retrieve the most relevant chunks from
  Chroma, and the LLM generates the final answer using both the retrieved
  context and the conversation history.
- The chat history is also mirrored in `st.session_state.chat_history` so it
  survives Streamlit's re-runs and is rendered back into the UI.
- Clicking **"Clear conversation"** resets both the LangChain memory object
  and the Streamlit session state, starting a fresh conversation without
  needing to re-upload or re-process the PDF... (re-processing is only
  required for a *new* PDF).

This allows natural follow-ups such as:

> **User:** What are the main findings of this paper?
> **Assistant:** [answer]
> **User:** Can you explain the second one in more detail?

---

## 🚧 Challenges Faced

- **Chunking trade-offs:** Too-small chunks lost surrounding context; too-large
  chunks diluted retrieval relevance and increased token usage. Settled on a
  configurable chunk size (default 1000 chars, 150 overlap) exposed as sliders
  in the sidebar.
- **Follow-up question resolution:** Naively re-running similarity search on
  each raw user message failed for short follow-ups (e.g. "what about that?").
  Solved by using the chain's built-in question-condensing step, which
  rewrites the query using chat history before retrieval.
- **Free-tier LLM constraints:** Some free LLM options have strict rate limits
  or require heavy local compute (e.g., self-hosted Ollama isn't available on
  Streamlit Cloud). Groq's hosted free API solved both the speed and
  deployability problem.
- **Python-version mismatch on Streamlit Cloud:** Deploying against Streamlit
  Community Cloud's newer default runtime (Python 3.13) crashed with
  `TypeError: 'function' object is not subscriptable` while resolving
  `Optional[dict[str, Any]]` in `langchain-core`. This is a known upstream
  incompatibility between older pinned LangChain/Pydantic internals and
  Python 3.13's typing changes. Fixed by adding a `.python-version` file
  pinning the deploy to **Python 3.11**, which Streamlit Cloud honors
  automatically.
- **ChromaDB + Streamlit Cloud's system SQLite:** Streamlit Community Cloud's
  base image ships an SQLite version older than Chroma requires, causing a
  runtime error on import. Fixed by installing `pysqlite3-binary` and
  swapping it into `sys.modules["sqlite3"]` at the very top of `app.py`,
  before Chroma is imported.
- **State management in Streamlit:** Since Streamlit reruns the whole script
  on every interaction, the vector store, chain, and memory all had to be
  persisted deliberately in `st.session_state` (and cached with
  `st.cache_resource` for the expensive PDF-processing step) to avoid
  reprocessing the PDF or losing conversation history on every keystroke.

---

## 🔮 Future Improvements

- Support multiple PDFs in a single session with source-document attribution
  per file.
- Persist the Chroma vector store to disk so re-opening the app doesn't
  require re-uploading the same PDF.
- Add a toggle to switch between multiple free LLM backends (Groq, Google
  Gemini free tier) for comparison.
- Stream tokens as they're generated for a more responsive typing effect.
- Add citation highlighting that jumps to the exact page/paragraph in an
  embedded PDF viewer.
- Add automated evaluation (e.g., RAGAS) to measure answer faithfulness and
  retrieval precision.

---

## 🛠️ Running Locally

```bash
git clone <your-repo-url>
cd pdf-rag-qa
pip install -r requirements.txt

# Add your free Groq API key (https://console.groq.com/keys)
export GROQ_API_KEY="your-key-here"   # or paste it directly in the sidebar

streamlit run app.py
```

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click
   **"New app"**.
3. Select this repo, branch `main`, and file `app.py`.
   (A `.python-version` file in the repo pins the runtime to Python 3.11 —
   important, since newer default runtimes on Streamlit Cloud can trigger a
   known LangChain/Pydantic typing bug. See "Challenges Faced" below.)
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click **Deploy**. Once live, paste the public URL at the top of this
   README and in your submission.
