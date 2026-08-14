from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.file_parser import download_and_parse   # ← was pdf_parser
from services.chunker import chunk_text
from services.embedder import embed_texts
from services.vector_store import store_chunks
from services.file_parser import download_file, parse_file, IMAGE_EXTENSIONS
from services.chunker import chunk_text_with_offsets
from services.topic_extractor import extract_topics, assign_topics_to_chunks
from services.embedder import embed_texts
from services.vector_store import store_chunks
import traceback


router = APIRouter()

class IngestRequest(BaseModel):
    pdf_url: str   # ← renamed from pdf_url (Node.js sends any file URL now)

@router.post("/process")
def process_file(req: IngestRequest):
    try:
        tmp_path, ext = download_file(req.pdf_url)
        text = parse_file(tmp_path, ext)

        if not text.strip():
            # e.g. Gemini found no extractable content in the image
            return {"status": "ok", "chunks_stored": 0}

        chunks     = chunk_text(text)
        embeddings = embed_texts(chunks)

        is_image = ext in IMAGE_EXTENSIONS
        metadata = {"file_type": "image", "page": 1} if is_image else None

        store_chunks(chunks, embeddings, req.pdf_url, metadata=metadata)

        return {"status": "ok", "chunks_stored": len(chunks)}
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topic")
def process_file(req: IngestRequest):
    try:
        tmp_path, ext = download_file(req.pdf_url)
        text = parse_file(tmp_path, ext)

        topics = extract_topics(tmp_path, ext, text)

        chunks_with_offsets = chunk_text_with_offsets(text)
        chunks = [c for c, _ in chunks_with_offsets]
        chunk_topics = assign_topics_to_chunks(chunks_with_offsets, topics)

        embeddings = embed_texts(chunks)
        store_chunks(chunks, embeddings, req.pdf_url, topics=chunk_topics)

        return {
            "status": "ok",
            "chunks_stored": len(chunks),
            "topics": [t.to_dict() for t in topics],
        }
    except ValueError as e: 
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))