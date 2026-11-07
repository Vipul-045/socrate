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
# Development defaults to socrate_docs_dev. Anywhere else the collection is
# named outright — production's is socrate_docs — and a development one is
# refused: a guess could point the service at an empty collection, and every
# document would look unindexed.
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "").strip()
if IS_DEVELOPMENT:
    QDRANT_COLLECTION = QDRANT_COLLECTION or "socrate_docs_dev"
elif not QDRANT_COLLECTION:
    raise RuntimeError("Refusing to start: QDRANT_COLLECTION is not set. Production's is socrate_docs; set it in the host's environment.")
elif QDRANT_COLLECTION.endswith("_dev"):
    raise RuntimeError(f"Refusing to start: QDRANT_COLLECTION is {QDRANT_COLLECTION}, a development collection. Production's is socrate_docs.")
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
# Set in development (MinIO, the backend's docker-compose.dev.yml): the bucket
# is read there, path-style, and nowhere else.
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
if IS_DEVELOPMENT and not (S3_ENDPOINT and is_local_url(S3_ENDPOINT)):
    raise RuntimeError(f"Refusing to start in development: S3_ENDPOINT is {S3_ENDPOINT or 'not set'}, so stored files would be read from AWS. Set it to the local MinIO in .env.development.")