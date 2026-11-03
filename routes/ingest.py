from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.ingest import ingest
import traceback


router = APIRouter()

class IngestRequest(BaseModel):
    pdf_url: str   # ← renamed from pdf_url (Node.js sends any file URL now)


def _ingest_or_http_error(url: str) -> dict:
    try:
        return ingest(url)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Both endpoints index the file once (services/ingest.py) and answer with its
# topics: whichever arrives first does the work, the other finds it done.
# text_status says whether there was text to index: "ok", "little_text" or
# "no_text" (a scanned page with no text layer, for instance).
@router.post("/process")
def process_file(req: IngestRequest):
    return _ingest_or_http_error(req.pdf_url)


@router.post("/topic")
def topics_for_file(req: IngestRequest):
    return _ingest_or_http_error(req.pdf_url)
