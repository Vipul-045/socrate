# Text recognition for PDFs without a text layer (scans, photographed pages).
# Each page is rendered to an image and Gemini transcribes it as Markdown —
# headings as #/##, tables as tables — so the document can be indexed and its
# topics found like any other. Pages are read in parallel; a long scan is read
# up to OCR_MAX_PAGES and the result says how many were.

import time
from concurrent.futures import ThreadPoolExecutor

# A page with fewer characters than this on average has no usable text layer.
MIN_CHARS_PER_PAGE = 40
OCR_MAX_PAGES = 60
OCR_WORKERS = 4

_OCR_PROMPT = """Transcribe ALL text on this page of a document, as Markdown.

- Preserve the text exactly as written, in reading order.
- Use markdown headings (#, ##, ###) for titles and section headers.
- Reproduce tables as markdown tables.
- Reproduce mathematical notation as LaTeX ($...$ inline, $$...$$ block).
- For a figure or diagram, add one line starting "**Figure:**" describing it.
- Do not add commentary, and do not wrap the output in code fences.
- If the page has no text, return nothing."""


def needs_ocr(path: str, text: str) -> bool:
    """A PDF whose pages hold (almost) no extractable text."""
    import fitz
    pages = max(1, len(fitz.open(path)))
    return len(text.strip()) / pages < MIN_CHARS_PER_PAGE


def _client():
    from google import genai
    from config import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set — cannot read scanned pages")
    return genai.Client(api_key=GEMINI_API_KEY)


def _read_page(client, png: bytes) -> str:
    from google.genai import types
    from google.genai.errors import ServerError
    for attempt in range(2):  # one retry
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[_OCR_PROMPT, types.Part.from_bytes(data=png, mime_type="image/png")],
                config=types.GenerateContentConfig(
                    temperature=0,
                    # Transcribing needs no reasoning: thinking would only add time and cost.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (response.text or "").strip()
        except ServerError:
            if attempt == 1:
                raise
            time.sleep(2)
    return ""


def ocr_pdf(path: str) -> dict:
    """{"text", "pages_read", "pages_total"}: the pages' text, page by page."""
    import fitz
    doc = fitz.open(path)
    total = len(doc)
    count = min(total, OCR_MAX_PAGES)
    images = [doc[i].get_pixmap(dpi=150, colorspace=fitz.csGRAY).tobytes("png") for i in range(count)]
    client = _client()
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        pages = list(pool.map(lambda png: _read_page(client, png), images))
    return {"text": "\n\n".join(p for p in pages if p), "pages_read": count, "pages_total": total}
