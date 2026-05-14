from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama

app = FastAPI(title="Clinical Insights API")

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your local Ollama setup
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text
CHAT_MODEL = "llama3.1:8b"  # change to any model shown by: ollama list

# ---------------------------------------------------------------------------
# ChromaDB — persistent local vector store
# ---------------------------------------------------------------------------
_embedding_fn = OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE_URL}/api/embeddings",
    model_name=EMBED_MODEL,
)

_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma_client.get_or_create_collection(
    name="clinical_docs",
    embedding_function=_embedding_fn,
)


# ---------------------------------------------------------------------------
# Helper: seed the collection with documents (call once at startup or via a
# separate ingestion script)
# ---------------------------------------------------------------------------
def ingest_documents(documents: list[str], ids: list[str]) -> None:
    """Add documents to the ChromaDB collection."""
    _collection.upsert(documents=documents, ids=ids)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str


# ---------------------------------------------------------------------------
# RAG streaming generator
# ---------------------------------------------------------------------------
async def generate_rag_response(query: str):
    try:
        # 1. Retrieve relevant context from ChromaDB
        results = _collection.query(query_texts=[query], n_results=3)
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs) if docs else "No relevant context found."

        # 2. Construct a strict RAG prompt
        system_prompt = (
            "You are a clinical research assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "If the answer is not in the context, say 'I do not have enough information'.\n\n"
            f"Context:\n{context}"
        )

        # 3. Stream response from the local Ollama LLM
        client = ollama.AsyncClient(host=OLLAMA_BASE_URL)
        stream = await client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
        )

        async for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield f"data: {content}\n\n"
                await asyncio.sleep(0)  # yield control to the event loop

    except Exception as e:
        # Errors inside a StreamingResponse generator cannot change the HTTP
        # status code (headers are already sent). Yield a final SSE error event
        # so the client knows something went wrong, then let the stream close.
        yield f"data: [ERROR] {e}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        return StreamingResponse(
            generate_rag_response(request.query), media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
