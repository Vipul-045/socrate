from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.embedder import embed_single
from services.vector_store import search_similar

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query_pdf(req: QueryRequest):
    try:
        query_vector   = embed_single(req.query)      # convert question to vector
        context_chunks = search_similar(query_vector) # find similar chunks
        return {"chunks": context_chunks}             # return raw chunks, no LLM
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))