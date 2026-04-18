from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.embedder import embed_single
from services.vector_store import search_similar
from services.llm import ask_llm_stream

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    file_url: str 

@router.post("/query")
def query_pdf(req: QueryRequest):
    try:
        query_vector   = embed_single(req.query)
        context_chunks = search_similar(query_vector, req.file_url) 

        return StreamingResponse(
            ask_llm_stream(req.query, context_chunks),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))