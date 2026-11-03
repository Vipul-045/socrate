# Loads all secrets from the environment so nothing is hardcoded.
#
# SOCRATE_ENV=development reads .env.development and never .env (which names
# the production Qdrant), and refuses to start unless Qdrant is on this
# machine. Anything else reads .env, as before. A variable already set in the
# environment wins over either file.

from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

IS_DEVELOPMENT = os.getenv("SOCRATE_ENV") == "development"
_ENV_FILE = Path(__file__).resolve().parent / (".env.development" if IS_DEVELOPMENT else ".env")
if IS_DEVELOPMENT and not _ENV_FILE.exists():
    raise RuntimeError(f"SOCRATE_ENV=development reads {_ENV_FILE}, which doesn't exist. Copy .env.development.example to it.")
load_dotenv(_ENV_FILE)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Unset, the collection is socrate_docs_dev on a local Qdrant and socrate_docs
# (production's) on any other, so a deploy that never set it keeps its data.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION") or ("socrate_docs_dev" if is_local_url(QDRANT_URL) else "socrate_docs")
if IS_DEVELOPMENT and not is_local_url(QDRANT_URL):
    raise RuntimeError(f"Refusing to start in development: QDRANT_URL is {QDRANT_URL}, not this machine. Fix it in .env.development.")
EMBEDDING_MODEL = "embed-english-v3.0"
LLM_MODEL = "llama-3.3-70b-versatile" 
CHUNK_SIZE        = 500   # characters per chunk
CHUNK_OVERLAP     = 50
TOPIC_AI_MODEL = os.getenv("COHERE_API_KEY")
TOPIC_AI_MAX_CHARS = 10000
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The app's private S3 bucket. Its objects are read with boto3, which takes
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the environment itself.
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")