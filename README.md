---
title: RAG Document QA
emoji: "📄"
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: false
license: mit
---

# RAG Document Q&A System

A production-ready Retrieval-Augmented Generation system that lets users upload documents and ask natural language questions with sourced, grounded answers.

[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/prakharprakarsh/rag-document-qa)

## Features

- Multi-format ingestion: PDF, TXT, Markdown
- 3 chunking strategies: Recursive, Character, Token with comparison metrics
- Hybrid search: Blends BM25 keyword matching with semantic vector search
- Source attribution: Every answer cites its source documents
- RAGAS evaluation: Measures faithfulness, relevancy, precision, and recall
- FastAPI backend: Clean REST API with OpenAPI docs
- Docker ready: One command to run everything

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6 |
| Vector DB | ChromaDB |
| LLM | Mistral-7B / GPT-4o-mini |
| Keyword Search | BM25 (rank-bm25) |
| API | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Containerization | Docker + Docker Compose |

## Quick Start

Clone and setup:

    git clone https://github.com/prakharprakarsh/rag-document-qa.git
    cd rag-document-qa
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Run:

    make run-api       # Terminal 1
    make run-frontend  # Terminal 2

With Docker:

    docker-compose up --build

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /upload | Upload and process a document |
| POST | /ask | Ask a question |
| POST | /clear | Clear the vector store |

## Evaluation Results (RAGAS)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.91 |
| Context Precision | 0.83 |
| Context Recall | 0.79 |

## Project Structure

    src/
      ingestion/       # Document loading, chunking, embedding
      retrieval/       # Vector store + hybrid search
      generation/      # LLM chain + prompts
      evaluation/      # RAGAS metrics
      api/             # FastAPI endpoints
    frontend/          # Streamlit UI
    Dockerfile
    docker-compose.yml
    Makefile

## License

MIT
