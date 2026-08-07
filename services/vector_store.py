from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType  # ← add this
)
import uuid
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
from qdrant_client.models import Filter, FieldCondition, MatchValue

def get_all_chunks(pdf_url: str) -> list[str]:
    all_chunks = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="pdf_url",
                        match=MatchValue(value=pdf_url)
                    )
                ]
            ),
            limit=100,            # page size — tune as needed
            offset=offset,
            with_payload=True,
            with_vectors=False,   # we don't need the vectors back
        )

        all_chunks.extend(points)

        if offset is None:
            break

    # sort by chunk_index so the summary reads in original document order
    all_chunks.sort(key=lambda p: p.payload["chunk_index"])
    return [p.payload["text"] for p in all_chunks]

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
        # ← create index on pdf_url so filtering works
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="pdf_url",
            field_schema=PayloadSchemaType.KEYWORD
        )

        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="topic",
            field_schema=PayloadSchemaType.KEYWORD
        )

def search_similar(query_vector: list[float], file_url: str, top_k: int = 5) -> list[str]:
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter={
            "must": [
                {
                    "key": "pdf_url",            # ← changed to pdf_url
                    "match": { "value": file_url }
                }
            ]
        }
    ).points
    return [hit.payload["text"] for hit in results]


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    pdf_url: str,
    topics: list[str | None] | None = None,
):
    ensure_collection()

    points = []

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = {
            "text": chunk,
            "pdf_url": pdf_url,
            "chunk_index": i,
        }

        if topics and i < len(topics) and topics[i]:
            payload["topic"] = topics[i]

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    BATCH_SIZE = 100
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
        )