# Loads all secrets from .env so nothing is hardcoded

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_COLLECTION = "socrate_docs"
EMBEDDING_MODEL = "embed-english-v3.0"
LLM_MODEL = "llama-3.3-70b-versatile" 
CHUNK_SIZE        = 500   # characters per chunk
CHUNK_OVERLAP     = 50
TOPIC_AI_MODEL = os.getenv("COHERE_API_KEY")
TOPIC_AI_MAX_CHARS = 10000