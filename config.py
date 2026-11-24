"""Central configuration for the RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking defaults
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ChromaDB
COLLECTION_NAME = "documents"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")  # "openai" or "huggingface"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# Model for the Hugging Face inference router. Deliberately a separate variable from
# LLM_MODEL_NAME (OpenAI), so an old value left in a Space's settings can't break it.
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Retrieval
TOP_K = 5
HYBRID_ALPHA = 0.5  # 0 = pure keyword, 1 = pure semantic

# API
API_HOST = "0.0.0.0"
API_PORT = 8000