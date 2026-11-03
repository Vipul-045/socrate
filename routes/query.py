from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.embedder import embed_single
from services.vector_store import search_similar, get_all_chunks, chunks_for_ranges
from services.llm import ask_llm_stream

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    pdf_url: str
    top_k: int = 5   # how many chunks; a large quiz asks for more (at most 60)

class StreamRequest(BaseModel):
    query: str
    chunks: list[str]  # chunks returned from /query/search

class FullDataRequest(BaseModel):
    pdf_url: str

class ContextRange(BaseModel):
    start: int
    end: int

class ContextRequest(BaseModel):
    pdf_url: str
    ranges: list[ContextRange] = []   # the selected topics' ranges; empty = whole document
    max_chars: int = 20000

@router.post("/query/full")
def get_full_data(req: FullDataRequest):
    try:
        all_chunks = get_all_chunks(req.pdf_url)
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No data found for this PDF")
        return { "chunks": all_chunks }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── The material a quiz, deck or exam is written from ────────────
# The selected topics' text (or the whole document's), spread evenly across
# each topic and sized to what was asked for, rather than the first pages.
@router.post("/query/context")
def context(req: ContextRequest):
    try:
        budget = max(1000, min(req.max_chars, 200000))
        return chunks_for_ranges(req.pdf_url, [(r.start, r.end) for r in req.ranges], budget)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 2: Semantic search only ──────────────────────────────
@router.post("/query/search")
def search(req: SearchRequest):
    try:
        query_vector   = embed_single(req.query)
        context_chunks = search_similar(query_vector, req.pdf_url, top_k=max(1, min(req.top_k, 60)))
        return { "chunks": context_chunks }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 3: LLM stream only ───────────────────────────────────
@router.post("/query/stream")
def stream(req: StreamRequest):
    try:
        return StreamingResponse(
            ask_llm_stream(req.query, req.chunks),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))