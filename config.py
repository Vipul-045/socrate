# Loads all secrets from .env so nothing is hardcoded

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "socrate_docs"
EMBEDDING_MODEL   = "text-embedding-3-small"
LLM_MODEL         = "gpt-4o"
CHUNK_SIZE        = 500   # characters per chunk
CHUNK_OVERLAP     = 50