# One function that detects the file type and routes to the right parser
# All parsers return plain text — the rest of the pipeline doesn't care about format

import requests
import tempfile
import os

# add near the top with your other imports
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".heic"}

_IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
}

_IMAGE_PROMPT = """Extract ALL content from this image and return it as structured Markdown.

- Preserve all visible text exactly as written.
- Use markdown headings (#, ##, ###) for titles/section headers.
- Reproduce any tables as markdown tables.
- For charts, diagrams, or figures, add a short plain-text description under a
  line that starts with "**Diagram description:**".
- Reproduce any mathematical notation as LaTeX ($...$ inline, $$...$$ block).
- Do not add commentary or wrap the output in code fences.
- If the image has no meaningful content, return an empty string."""


import time

def _parse_image(path: str) -> str:
    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError
    from config import GEMINI_API_KEY

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set — cannot parse image files")

    client = genai.Client(api_key=GEMINI_API_KEY)

    ext = os.path.splitext(path)[-1].lower()
    mime_type = _IMAGE_MIME_MAP.get(ext, "image/png")

    with open(path, "rb") as f:
        image_bytes = f.read()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    _IMAGE_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return (response.text or "").strip()
        except ServerError as e:
            if attempt == max_retries - 1:
                raise ValueError(
                    "Gemini is currently unavailable (high demand). Please try again shortly."
                ) from e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff

    return (response.text or "").strip()

def download_and_parse(pdf_url: str) -> str:
    # 1. Download the file
    response = requests.get(pdf_url)
    response.raise_for_status()

    # 2. Detect extension from the URL
    ext = os.path.splitext(pdf_url.split("?")[0])[-1].lower()

    # 3. Save to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(response.content)
        tmp_path = f.name

    # 4. Route to the right parser based on extension
    parsers = {
        ".pdf":  _parse_pdf,
        ".txt":  _parse_txt,
        ".md":   _parse_txt,   # markdown is just text
        ".docx": _parse_docx,
        ".doc":  _parse_docx,  # treated same as docx
        ".xlsx": _parse_xlsx,
        ".xls":  _parse_xlsx,
        ".pptx": _parse_pptx,
        ".ppt":  _parse_pptx,
        ".rtf":  _parse_rtf,
        ".csv":  _parse_txt, 
        ".png":  _parse_image,   # ← new
        ".jpg":  _parse_image,   # ← new
        ".jpeg": _parse_image,   # ← new
        ".webp": _parse_image,   # ← new
        ".heic": _parse_image,   # csv is plain text
    }

    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")

    return parser(tmp_path)


# ── Individual parsers ──────────────────────────────────────────────

def _parse_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _parse_txt(path: str) -> str:
    # Plain text, markdown, CSV — just read it
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    # Each paragraph is one line; join them all
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_xlsx(path: str) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    lines = []
    for sheet in wb.worksheets:
        lines.append(f"[Sheet: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            # Convert each row to tab-separated text, skip fully empty rows
            row_text = "\t".join(str(c) if c is not None else "" for c in row)
            if row_text.strip():
                lines.append(row_text)
    return "\n".join(lines)


def _parse_pptx(path: str) -> str:
    from pptx import Presentation
    prs = Presentation(path)
    lines = []
    for i, slide in enumerate(prs.slides):
        lines.append(f"[Slide {i+1}]")
        for shape in slide.shapes:
            # Only text-bearing shapes (titles, text boxes, tables)
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        lines.append(text)
    return "\n".join(lines)


def _parse_rtf(path: str) -> str:
    from striprtf.striprtf import rtf_to_text
    with open(path, "r", errors="ignore") as f:
        return rtf_to_text(f.read())

        # add these two functions, keep everything else (the parsers) unchanged

def download_file(pdf_url: str) -> tuple[str, str]:
    response = requests.get(pdf_url)
    response.raise_for_status()
    ext = os.path.splitext(pdf_url.split("?")[0])[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(response.content)
        tmp_path = f.name
    return tmp_path, ext


_PARSERS = {
    ".pdf": _parse_pdf, ".txt": _parse_txt, ".md": _parse_txt,
    ".docx": _parse_docx, ".doc": _parse_docx,
    ".xlsx": _parse_xlsx, ".xls": _parse_xlsx,
    ".pptx": _parse_pptx, ".ppt": _parse_pptx,
    ".rtf": _parse_rtf, ".csv": _parse_txt,
     ".png": _parse_image, ".jpg": _parse_image, ".jpeg": _parse_image,
    ".webp": _parse_image, ".heic": _parse_image,
}


def parse_file(tmp_path: str, ext: str) -> str:
    parser = _PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(tmp_path)


def download_and_parse(pdf_url: str) -> str:
    """Kept for backward compatibility with any other callers."""
    tmp_path, ext = download_file(pdf_url)
    return parse_file(tmp_path, ext)