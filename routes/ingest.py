from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.file_parser import download_and_parse   # ← was pdf_parser
from services.chunker import chunk_text
from services.embedder import embed_texts
from services.vector_store import store_chunks

router = APIRouter()

class IngestRequest(BaseModel):
    pdf_url: str   # ← renamed from pdf_url (Node.js sends any file URL now)

@router.post("/process")
def process_file(req: IngestRequest):
    try:
        text       = download_and_parse(req.pdf_url)
        chunks     = chunk_text(text)
        embeddings = embed_texts(chunks)
        store_chunks(chunks, embeddings, req.pdf_url)
        return {"status": "ok", "chunks_stored": len(chunks)}
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))  # 415 = Unsupported Media Type
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))