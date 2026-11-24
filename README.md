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

# RAG Document Q&A

Upload a PDF, TXT or Markdown file and ask questions about it. Answers cite the chunks they came from. Retrieval blends BM25 keyword scores with vector similarity.

[![CI](https://github.com/prakharprakarsh/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/prakharprakarsh/rag-document-qa/actions/workflows/ci.yml)

**Live demo:** https://huggingface.co/spaces/prakharprakarsh/rag-document-qa

## How it works

```
upload ─► load (pypdf / text) ─► chunk ─► embed (MiniLM-L6) ─► ChromaDB
                                                                   │
question ─► BM25 over all chunks ──┐                               │
         └► vector top-20 ─────────┴─► blend (alpha) ─► top-k ─► prompt ─► LLM ─► answer + sources
```

- **Chunking:** recursive (default), character or token splitter, configurable size/overlap. Each chunk keeps its source file and page.
- **Hybrid search:** `score = alpha · semantic + (1 − alpha) · BM25/max(BM25)`, alpha = 0.5 by default. The semantic side pulls a wider candidate pool (`max(4k, 20)`) so a chunk that's mid-ranked semantically but top on keywords can still make the cut.
- **Generation:** any OpenAI-compatible chat endpoint. If `GROQ_API_KEY` is set, Groq's free tier is tried first (`openai/gpt-oss-120b`, then Llama 3.3 70B); otherwise or on failure, the Hugging Face router. No provider is pinned on HF, and each provider falls back through a short model list. If no model can answer (no key, no credits, rate limit), the app shows the most relevant passages instead of an error. The prompt tells the model to answer only from the provided context.

## Things I fixed along the way

- "Hybrid" was silently pure vector search: neither the API nor the Streamlit app ever built a `HybridSearcher`, so the code fell back. `ask_question` now builds one from the stored chunks.
- BM25 tokenised with `str.split()`, so `rent,` and `rent` were different terms. Now uses `\w+`.
- Hugging Face's free inference credits ran out (HTTP 402) and the app just printed the error. It now supports Groq's free tier and falls back to showing the retrieved passages when no model is reachable.
- The model was pinned to one provider (`:cerebras`). When Cerebras retired Llama 3.1 8B, every answer became a 410 error. The app now lets the router choose a provider and falls back through a short list of models.
- Building the token splitter downloads the tiktoken vocabulary, and the code built all three splitters up front, so choosing *recursive* failed offline. Splitters are now built on demand.

## Run it

```bash
git clone https://github.com/prakharprakarsh/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add GROQ_API_KEY (free at console.groq.com) and/or HUGGINGFACEHUB_API_TOKEN

streamlit run app.py                    # single-process app (what the HF Space runs)
# or API + separate frontend:
make run-api                            # FastAPI on :8000, docs at /docs
make run-frontend
```

Docker: `docker-compose up --build`.

## API

| Method | Endpoint | |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/upload` | Upload and index a document |
| POST | `/ask` | `{question, search_type: hybrid\|semantic, top_k}` → answer + sources |
| POST | `/clear` | Empty the vector store |

## Tests

```bash
pip install -r requirements-test.txt
pytest -q
```

Covers loading, chunking, the hybrid score blend (alpha = 0 / 0.5 / 1), tokenisation and answer assembly. The vector store and LLM are stubbed, so no GPU or API key is needed.

## Evaluation

`src/evaluation/ragas_eval.py` runs RAGAS (faithfulness, answer relevancy, context precision and recall) over a list of questions with reference answers. I haven't committed a question set or results yet. The next step is a small set of Q&A pairs on a public document (e.g. a Dutch rental contract template) to compare hybrid vs. semantic retrieval and the chunking strategies.

## Layout

```
src/
  ingestion/   loader, chunker, embedder
  retrieval/   ChromaDB wrapper, hybrid search
  generation/  prompt, LLM call, answer + sources
  evaluation/  RAGAS runner
  api/         FastAPI
frontend/      Streamlit client for the API
app.py         single-process Streamlit app (HF Spaces)
```

## License

MIT — Prakhar Prakarsh
