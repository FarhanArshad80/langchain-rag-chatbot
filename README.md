# Agentic RAG Chatbot with Local LLM (Ollama & LangGraph)

Upload a PDF and ask questions about it. A LangGraph agent retrieves the relevant parts of the document and a local LLM generates the answer using only that content.

## Working Flow

```
PDF upload -> Load pages -> Split into chunks -> Embeddings -> Chroma DB
                                                                  |
Question -> [LangGraph Agent]                                     |
              |                                                   |
              +-> retrieve node (top 2 chunks) <------------------+
              |
              +-> generate node (Prompt -> LLM) -> Answer
```

1. **Upload**: the PDF is saved to a temporary file.
2. **Load**: `PyPDFLoader` reads the PDF page by page.
3. **Split**: text is cut into chunks (1000 chars, 200 overlap).
4. **Embed and store**: `nomic-embed-text` converts chunks to vectors, stored in Chroma.
5. **Agent workflow**: a LangGraph `StateGraph` runs the question through two nodes, sharing a typed state (`question`, `context`, `generation`, `decision`).
   - **retrieve**: fetches the 2 chunks most similar to the question.
   - **generate**: passes the chunks and question to the LLM. If the document does not contain the answer, the agent replies that it cannot find it.

## Setup

```bash
pip install streamlit python-dotenv pypdf chromadb langchain-core langchain-community langchain-text-splitters langchain-ollama langchain-chroma langgraph

ollama pull nomic-embed-text
ollama pull llama3.2
```

Optional `.env` (for LangSmith tracing only):

```
LANGCHAIN_API_KEY=your_key
```

## Run

```bash
streamlit run app.py
```

## Configuration

| Setting | Default |
|---|---|
| Chat model | `llama3.2` |
| Embedding model | `nomic-embed-text` |
| Chunk size / overlap | `1000` / `200` |
| Retrieved chunks (`k`) | `2` |
| Temperature | `0` |

## Notes

- The current graph is linear (retrieve, then generate). The `decision` field in the state is reserved for future conditional routing.
- The vector store is in-memory and is rebuilt when the app restarts.
- Smaller models such as `qwen:0.5b` also work but give weaker answers; `llama3.2` or `qwen2.5:3b` are recommended.

## Roadmap

- Document grading node with conditional edges
- Query rewriting when retrieval is poor
- Source citations with page numbers