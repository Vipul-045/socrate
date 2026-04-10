# Node.js hits POST /query with the user's question
# We run steps 6 → 7 and return the answer

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.embedder import embed_single
from services.vector_store import search_similar
from services.llm import ask_llm

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query_pdf(req: QueryRequest):
    try:
        query_vector   = embed_single(req.query)          # Step 6a: embed the query
        context_chunks = search_similar(query_vector)     # Step 6b: find similar chunks
        answer         = ask_llm(req.query, context_chunks)  # Step 7: call LLM
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))