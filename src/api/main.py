"""FastAPI backend for the RAG Document Q&A system."""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from src.api.schemas import (
    QuestionRequest, AnswerResponse, UploadResponse, HealthResponse,
)
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import add_documents, clear_vector_store
from src.generation.llm_chain import ask_question

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload documents and ask questions using RAG with hybrid search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
):
    """Upload a document and add it to the vector store."""
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Load and chunk
        documents = load_document(tmp_path)
        # Update source metadata to use original filename
        for doc in documents:
            doc.metadata["source"] = file.filename
        chunks = chunk_documents(documents, strategy=chunk_strategy, chunk_size=chunk_size)
        # Store in vector DB
        add_documents(chunks)

        return UploadResponse(
            message="Document processed successfully",
            num_chunks=len(chunks),
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Ask a question about the uploaded documents."""
    try:
        result = ask_question(
            question=request.question,
            search_type=request.search_type,
            k=request.top_k,
        )
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            search_type=request.search_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear():
    """Clear all documents from the vector store."""
    clear_vector_store()
    return {"message": "Vector store cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)