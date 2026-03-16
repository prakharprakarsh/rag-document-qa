# 📄 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation system that lets users upload documents and ask natural language questions with sourced, grounded answers.

🔗 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/rag-document-qa)**

## Architecture
```
┌─────────────┐     ┌─────────────────────────────────────────────┐
│   User       │     │              RAG Pipeline                    │
│  (Streamlit) │────▶│                                             │
│              │     │  ┌───────────┐   ┌──────────────────────┐  │
└─────────────┘     │  │  Loader    │──▶│  Chunker             │  │
                    │  │  (PDF/TXT) │   │  (Recursive/Char/    │  │
                    │  └───────────┘   │   Token strategies)   │  │
                    │                   └──────────┬───────────┘  │
                    │                              ▼              │
                    │                   ┌──────────────────────┐  │
                    │                   │  Embedder            │  │
                    │                   │  (MiniLM-L6-v2)      │  │
                    │                   └──────────┬───────────┘  │
                    │                              ▼              │
                    │  ┌───────────┐   ┌──────────────────────┐  │
                    │  │  BM25     │──▶│  Hybrid Search        │  │
                    │  │ (Keyword) │   │  (α blend)            │  │
                    │  └───────────┘   └──────────┬───────────┘  │
                    │                              ▼              │
                    │  ┌───────────┐   ┌──────────────────────┐  │
                    │  │ ChromaDB  │──▶│  LLM Generation       │  │
                    │  │ (Vectors) │   │  (Mistral/GPT-4o)     │  │
                    │  └───────────┘   └──────────┬───────────┘  │
                    │                              ▼              │
                    │                   ┌──────────────────────┐  │
                    │                   │  Answer + Sources     │  │
                    │                   └──────────────────────┘  │
                    └─────────────────────────────────────────────┘
```

## Features

- **Multi-format ingestion**: PDF, TXT, Markdown
- **3 chunking strategies**: Recursive, Character, Token — with comparison metrics
- **Hybrid search**: Blends BM25 keyword matching with semantic vector search
- **Source attribution**: Every answer cites its source documents
- **RAGAS evaluation**: Measures faithfulness, relevancy, precision, and recall
- **FastAPI backend**: Clean REST API with OpenAPI docs
- **Docker ready**: One command to run everything

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Framework        | LangChain                           |
| Embeddings       | sentence-transformers/all-MiniLM-L6 |
| Vector DB        | ChromaDB                            |
| LLM              | Mistral-7B / GPT-4o-mini            |
| Keyword Search   | BM25 (rank-bm25)                    |
| API              | FastAPI                             |
| Frontend         | Streamlit                           |
| Evaluation       | RAGAS                               |
| Containerization | Docker + Docker Compose             |

## Quick Start
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Edit with your API keys

# Run
make run-api       # Terminal 1: starts FastAPI on :8000
make run-frontend  # Terminal 2: starts Streamlit on :8501
```

### With Docker
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint  | Description                  |
|--------|-----------|------------------------------|
| GET    | /health   | Health check                 |
| POST   | /upload   | Upload and process a document|
| POST   | /ask      | Ask a question               |
| POST   | /clear    | Clear the vector store       |

Full API docs available at `http://localhost:8000/docs` when running.

## Chunking Strategy Comparison

| Strategy  | Chunks | Avg Length | Best For                    |
|-----------|--------|------------|-----------------------------|
| Recursive | ~45    | 480 chars  | General documents (default) |
| Character | ~38    | 510 chars  | Well-structured text        |
| Token     | ~52    | 450 chars  | Precise token control       |

*(Run on a sample 25-page PDF. Your results will vary.)*

## Evaluation Results (RAGAS)

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 0.87  |
| Answer Relevancy  | 0.91  |
| Context Precision | 0.83  |
| Context Recall    | 0.79  |

*(Evaluated on a 20-question test set. Run `python -m src.evaluation.ragas_eval` to reproduce.)*

## Project Structure
```
├── src/
│   ├── ingestion/       # Document loading, chunking, embedding
│   ├── retrieval/       # Vector store + hybrid search
│   ├── generation/      # LLM chain + prompts
│   ├── evaluation/      # RAGAS metrics
│   └── api/             # FastAPI endpoints
├── frontend/            # Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

MIT
