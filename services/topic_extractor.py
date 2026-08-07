import re
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from config import COHERE_API_KEY

# --- ADD THESE TO config.py ---
# TOPIC_AI_MODEL = "command-r-plus"
# TOPIC_AI_MAX_CHARS = 12000
from config import TOPIC_AI_MODEL, TOPIC_AI_MAX_CHARS


@dataclass
class Topic:
    id: str
    title: str
    level: int
    startPage: Optional[int] = None
    endPage: Optional[int] = None
    # internal-only, used to map chunks -> topics, not part of the public shape
    char_start: Optional[int] = field(default=None, repr=False)
    char_end: Optional[int] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "startPage": self.startPage,
            "endPage": self.endPage,
        }


def _fill_end_pages_and_offsets(topics: list[Topic], total_pages: Optional[int] = None):
    for i, t in enumerate(topics):
        if i + 1 < len(topics):
            t.char_end = topics[i + 1].char_start
            if t.startPage is not None and topics[i + 1].startPage is not None:
                t.endPage = topics[i + 1].startPage
        else:
            t.char_end = None
            t.endPage = total_pages


# ────────────────────────── PDF ──────────────────────────

def _pdf_page_char_offsets(doc) -> dict[int, int]:
    """Mirrors '\\n'.join(page.get_text() for page in doc) from file_parser._parse_pdf"""
    offsets = {}
    cursor = 0
    for i, page in enumerate(doc):
        offsets[i + 1] = cursor
        cursor += len(page.get_text()) + 1
    return offsets


def _extract_topics_pdf_toc(doc) -> list[Topic]:
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...], 1-indexed pages
    if not toc:
        return []
    page_offsets = _pdf_page_char_offsets(doc)
    topics = []
    for level, title, page in toc:
        topics.append(Topic(
            id=str(uuid.uuid4()),
            title=title.strip(),
            level=level,
            startPage=page,
            char_start=page_offsets.get(page, 0),
        ))
    _fill_end_pages_and_offsets(topics, total_pages=doc.page_count)
    return topics


def _extract_topics_pdf_heuristic(doc) -> list[Topic]:
    """Fallback when there's no outline: font size + bold heuristics."""
    page_offsets = _pdf_page_char_offsets(doc)
    spans = []
    size_counter = Counter()

    for i, page in enumerate(doc):
        page_num = i + 1
        cursor = 0
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", []))
                if not line_text.strip():
                    cursor += len(line_text) + 1
                    continue
                sizes = [s["size"] for s in line["spans"]]
                bold = any(s["flags"] & (1 << 4) for s in line["spans"])
                size = max(sizes) if sizes else 0
                size_counter[round(size, 1)] += 1
                spans.append({
                    "page": page_num, "text": line_text.strip(),
                    "size": size, "bold": bold, "offset_in_page": cursor,
                })
                cursor += len(line_text) + 1

    if not spans:
        return []

    body_size = size_counter.most_common(1)[0][0]
    heading_sizes = sorted(
        {round(s["size"], 1) for s in spans if s["size"] > body_size * 1.1 or s["bold"]},
        reverse=True,
    )
    if not heading_sizes:
        return []

    size_to_level = {sz: idx + 1 for idx, sz in enumerate(heading_sizes[:3])}

    def level_for(size, bold):
        sz = round(size, 1)
        if sz in size_to_level:
            return size_to_level[sz]
        return 3 if bold else None

    topics = []
    for s in spans:
        if len(s["text"]) > 120:
            continue  # skip full paragraphs, only short heading-like lines
        lvl = level_for(s["size"], s["bold"])
        if lvl is None:
            continue
        topics.append(Topic(
            id=str(uuid.uuid4()), title=s["text"], level=lvl,
            startPage=s["page"], char_start=page_offsets[s["page"]] + s["offset_in_page"],
        ))

    _fill_end_pages_and_offsets(topics, total_pages=doc.page_count)
    return topics


# ────────────────────────── DOCX ──────────────────────────

def _looks_bold_heading(paragraph) -> bool:
    runs = [r for r in paragraph.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def _extract_topics_docx(path: str) -> list[Topic]:
    from docx import Document
    doc = Document(path)

    heading_style_re = re.compile(r"^(Heading|Title)\s*(\d*)$", re.IGNORECASE)
    numbered_re = re.compile(r"^(\d+(\.\d+)*)\s+\S")

    topics = []
    cursor = 0
    # must mirror file_parser._parse_docx: "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        style_name = (p.style.name or "") if p.style else ""
        m = heading_style_re.match(style_name)
        level = None
        if m:
            level = int(m.group(2)) if m.group(2) else 1
        elif numbered_re.match(text) and len(text) < 100:
            level = text.split()[0].count(".") + 1
        elif _looks_bold_heading(p) and len(text) < 100:
            level = 2

        if level:
            topics.append(Topic(id=str(uuid.uuid4()), title=text, level=level, char_start=cursor))

        cursor += len(text) + 1

    _fill_end_pages_and_offsets(topics)  # no page concept in docx
    return topics


# ────────────────────────── TXT / MD ──────────────────────────

def _extract_topics_text(text: str) -> list[Topic]:
    md_re = re.compile(r"^(#{1,6})\s+(.*)")
    numbered_re = re.compile(r"^(\d+(\.\d+)*)\s+\S")

    topics = []
    cursor = 0
    for line in text.split("\n"):
        stripped = line.strip()
        m = md_re.match(stripped)
        if m:
            topics.append(Topic(
                id=str(uuid.uuid4()), title=m.group(2).strip(),
                level=len(m.group(1)), char_start=cursor,
            ))
        elif numbered_re.match(stripped) and 0 < len(stripped) < 100 and stripped[-1] not in ".,;:":
            topics.append(Topic(
                id=str(uuid.uuid4()), title=stripped,
                level=stripped.split()[0].count(".") + 1, char_start=cursor,
            ))
        cursor += len(line) + 1

    _fill_end_pages_and_offsets(topics)
    return topics


# ────────────────────────── AI fallback ──────────────────────────

def _extract_topics_ai(text: str) -> list[Topic]:
    """Only reached when no deterministic structure was found."""
    if not text.strip():
        return []

    sample = text[:TOPIC_AI_MAX_CHARS]
    prompt = (
        "You are analyzing a document to identify its topic/section structure. "
        "Return ONLY a JSON array (no prose, no markdown fences) of objects: "
        '{"title": string, "level": integer (1=top level), '
        '"snippet": string (first ~6 words of that section, copied exactly from the text)}. '
        "List topics in document order. If there's no clear structure, return one topic "
        "covering the whole document.\n\nTEXT:\n" + sample
    )

    try:
        import cohere
        client = cohere.Client(api_key=COHERE_API_KEY)
        resp = client.chat(message=prompt, model=TOPIC_AI_MODEL, temperature=0)
        raw = re.sub(r"^```(json)?|```$", "", resp.text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
    except Exception:
        # last-resort: whole doc as one topic, never fail ingestion because of this
        return [Topic(id=str(uuid.uuid4()), title="Document", level=1, char_start=0, char_end=len(text))]

    topics = []
    for item in parsed:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        level = int(item.get("level") or 1)
        snippet = str(item.get("snippet", "")).strip()
        char_start = text.find(snippet) if snippet else -1
        topics.append(Topic(
            id=str(uuid.uuid4()), title=title, level=level,
            char_start=char_start if char_start != -1 else None,
        ))

    located = [t for t in topics if t.char_start is not None]
    located.sort(key=lambda t: t.char_start)
    _fill_end_pages_and_offsets(located)
    return located or topics


# ────────────────────────── dispatcher ──────────────────────────

def extract_topics(file_path: str, ext: str, text: str) -> list[Topic]:
    ext = ext.lower()
    topics: list[Topic] = []
    try:
        if ext == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            topics = _extract_topics_pdf_toc(doc) or _extract_topics_pdf_heuristic(doc)
        elif ext in (".docx", ".doc"):
            topics = _extract_topics_docx(file_path)
        elif ext in (".txt", ".md", ".csv"):
            topics = _extract_topics_text(text)
    except Exception:
        topics = []

    if not topics:
        topics = _extract_topics_ai(text)

    return topics


def assign_topics_to_chunks(
    chunks_with_offsets: list[tuple[str, int]], topics: list[Topic]
) -> list[Optional[str]]:
    """Tags each chunk with the title of the topic it falls under, by char offset."""
    located = sorted([t for t in topics if t.char_start is not None], key=lambda t: t.char_start)
    if not located:
        return [None] * len(chunks_with_offsets)

    result = []
    for _, offset in chunks_with_offsets:
        title = None
        for t in located:
            if t.char_start <= offset and (t.char_end is None or offset < t.char_end):
                title = t.title
        result.append(title)
    return result