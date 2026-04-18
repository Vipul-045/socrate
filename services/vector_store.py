# Handles both storing chunks and searching them later
# Qdrant stores each chunk as a "point": vector + payload (metadata)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
import uuid
from config import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def ensure_collection():
    # Create the collection only if it doesn't exist yet
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=1024,           # dimension for text-embedding-3-small
                distance=Distance.COSINE
            )
        )

def store_chunks(chunks: list[str], embeddings: list[list[float]], pdf_url: str):
    ensure_collection()

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),   # unique ID for each chunk
            vector=vector,
            payload={
                "text": chunk,      # the actual text (returned at search time)
                "pdf_url": pdf_url,
                "chunk_index": i
            }
        ))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

def search_similar(query_vector: list[float], file_url: str, top_k: int = 5) -> list[str]:
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter={
            "must": [
                {
                    "key": "file_url",
                    "match": { "value": file_url }  # ← only search this file's chunks
                }
            ]
        }
    ).points

    return [hit.payload["text"] for hit in results]