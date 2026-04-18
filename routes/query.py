from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.embedder import embed_single
from services.vector_store import search_similar
from services.llm import ask_llm_stream

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    pdf_url: str

class StreamRequest(BaseModel):
    query: str
    chunks: list[str]  # chunks returned from /query/search


# ── Endpoint 1: Semantic search only ──────────────────────────────
@router.post("/query/search")
def search(req: SearchRequest):
    try:
        query_vector   = embed_single(req.query)
        context_chunks = search_similar(query_vector, req.pdf_url)
        return { "chunks": context_chunks }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 2: LLM stream only ───────────────────────────────────
@router.post("/query/stream")
def stream(req: StreamRequest):
    try:
        return StreamingResponse(
            ask_llm_stream(req.query, req.chunks),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))