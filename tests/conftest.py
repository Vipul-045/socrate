import os
import sys

# Tests import the service's modules from the repo root, and never read a
# .env: config.py would otherwise load one and refuse or reach real services.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_COLLECTION", "test_docs")
os.environ.setdefault("COHERE_API_KEY", "test")
