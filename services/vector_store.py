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

    # sort by chunk_index so the summary reads in original document order.
    # Documents uploaded before ingest ran once were stored up to three times:
    # each chunk is returned once.
    all_chunks.sort(key=lambda p: p.payload["chunk_index"])
    seen, unique = set(), []
    for p in all_chunks:
        if p.payload["chunk_index"] in seen:
            continue
        seen.add(p.payload["chunk_index"])
        unique.append(p.payload["text"])
    return unique

_ensured = False


def ensure_collection():
    global _ensured
    if _ensured:
        return
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
    field_name="image",
    field_schema=PayloadSchemaType.KEYWORD
)

        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="topic",
            field_schema=PayloadSchemaType.KEYWORD
        )
    # Added after the collection first existed: created here for an existing
    # collection too (creating an index that exists is a no-op).
    for field, schema in (("content_hash", PayloadSchemaType.KEYWORD),
                          ("char_start", PayloadSchemaType.INTEGER),
                          ("chunk_index", PayloadSchemaType.INTEGER)):
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name=field, field_schema=schema)
    _ensured = True


def _url_filter(pdf_url: str) -> Filter:
    return Filter(must=[FieldCondition(key="pdf_url", match=MatchValue(value=pdf_url))])


def count_points(pdf_url: str) -> int:
    """How many chunks are stored for this file (0 if none, or no collection yet)."""
    ensure_collection()
    return client.count(collection_name=QDRANT_COLLECTION, count_filter=_url_filter(pdf_url), exact=True).count


def copy_points_for_hash(content_hash: str, pdf_url: str) -> int:
    """Stores, under `pdf_url`, the chunks of an earlier upload of the same
    bytes (same SHA-256): its vectors are reused, nothing is embedded. Returns
    how many were copied (0 if no earlier upload is indexed)."""
    ensure_collection()
    hash_filter = Filter(must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))])
    first, _ = client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=hash_filter, limit=1, with_payload=True)
    if not first:
        return 0
    source = first[0].payload["pdf_url"]
    copied, offset = 0, None
    while True:
        points, offset = client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=_url_filter(source),
                                       limit=256, offset=offset, with_payload=True, with_vectors=True)
        if points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=[
                PointStruct(id=str(uuid.uuid4()), vector=p.vector, payload={**p.payload, "pdf_url": pdf_url})
                for p in points
            ])
            copied += len(points)
        if offset is None:
            return copied

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
    # Documents stored before ingest ran once hold each chunk up to three
    # times: the same text is returned once.
    seen, texts = set(), []
    for hit in results:
        if hit.payload["text"] not in seen:
            seen.add(hit.payload["text"])
            texts.append(hit.payload["text"])
    return texts


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    pdf_url: str,
    topics: list[str | None] | None = None,
    metadata: dict | None = None,          # ← new
    offsets: list[int] | None = None,      # each chunk's start in the parsed text
    content_hash: str | None = None,       # SHA-256 of the file, for re-uploads
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

        if offsets and i < len(offsets):
            payload["char_start"] = offsets[i]

        if content_hash:
            payload["content_hash"] = content_hash

        if metadata:                        # ← new — e.g. {"file_type": "image", "page": 1}
            payload.update(metadata)

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

def _even_sample(items: list, budget_chars: int) -> list:
    """Items (in document order) spread evenly across the whole list, until
    their text fills `budget_chars`: a range larger than its budget is
    covered from start to end, not cut off after its first pages."""
    total = sum(len(i["text"]) for i in items)
    if total <= budget_chars:
        return items
    keep = max(1, int(len(items) * budget_chars / total))
    step = len(items) / keep
    return [items[int(k * step)] for k in range(keep)]


def chunks_for_ranges(pdf_url: str, ranges: list[tuple[int, int]] | None, budget_chars: int) -> dict:
    """The document's chunks within the given character ranges (each topic's
    range), or the whole document when `ranges` is empty, sampled evenly to
    fit `budget_chars` overall. The budget is split across ranges in
    proportion to their size. Each chunk appears once, in document order.

    Returns {"chunks": [{text, char_start, chunk_index, range}],
             "ranges": [{"available_chars", "returned_chars"}],
             "located": bool}. located is False for a document stored before
    chunk offsets were kept: ranges can't be applied to it, and the caller
    falls back to search."""
    ensure_collection()
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=QDRANT_COLLECTION, scroll_filter=_url_filter(pdf_url),
                                      limit=256, offset=offset, with_payload=True, with_vectors=False)
        points.extend(batch)
        if offset is None:
            break
    seen, chunks = set(), []
    for p in sorted(points, key=lambda p: p.payload.get("chunk_index", 0)):
        idx = p.payload.get("chunk_index")
        if idx in seen:
            continue
        seen.add(idx)
        chunks.append({"text": p.payload["text"], "char_start": p.payload.get("char_start"), "chunk_index": idx})
    located = bool(chunks) and all(c["char_start"] is not None for c in chunks)
    if not ranges or not located:
        picked = _even_sample(chunks, budget_chars)
        return {"chunks": [{**c, "range": None} for c in picked],
                "ranges": [], "located": located or not ranges}

    groups = []
    for start, end in ranges:
        groups.append([c for c in chunks if start <= c["char_start"] < end])
    sizes = [sum(len(c["text"]) for c in g) for g in groups]
    total = sum(sizes) or 1
    picked, report = [], []
    for i, g in enumerate(groups):
        share = int(budget_chars * sizes[i] / total)
        chosen = _even_sample(g, max(share, 1)) if g else []
        picked.extend({**c, "range": i} for c in chosen)
        report.append({"available_chars": sizes[i], "returned_chars": sum(len(c["text"]) for c in chosen)})
    unique, seen = [], set()
    for c in sorted(picked, key=lambda c: c["char_start"]):
        if c["chunk_index"] not in seen:
            seen.add(c["chunk_index"])
            unique.append(c)
    return {"chunks": unique, "ranges": report, "located": True}
