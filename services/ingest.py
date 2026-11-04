# One pass over an uploaded file: download, parse, find its topics, chunk,
# embed, store. /process and /topic both come here, so a document is embedded
# and stored once however many times either is called:
#   - a URL that is already indexed is not embedded again (only its topics are
#     read back from the file, which takes well under a second);
#   - the same bytes uploaded again under a new URL reuse the stored vectors
#     (matched by SHA-256), without calling the embedding API.

import hashlib
import threading
from contextlib import contextmanager

from services.file_parser import download_file, parse_file, IMAGE_EXTENSIONS
from services.chunker import chunk_text_with_offsets
from services.topic_extractor import extract_topics, assign_topics_to_chunks
from services.embedder import embed_texts
from services import vector_store
from services.ocr import needs_ocr, ocr_pdf
import traceback

# Below this, a document is "very little text": still indexed, but the app
# says so instead of promising a full quiz from it.
LITTLE_TEXT_CHARS = 400

# Two requests for the same URL at once (the upload's /process and a quiz
# screen's /topic) wait for one ingest instead of embedding twice. Each URL's
# lock is dropped once nobody holds or waits for it.
_locks: dict[str, list] = {}  # url -> [lock, holders and waiters]
_locks_guard = threading.Lock()


@contextmanager
def _one_at_a_time(url: str):
    with _locks_guard:
        entry = _locks.setdefault(url, [threading.Lock(), 0])
        entry[1] += 1
    entry[0].acquire()
    try:
        yield
    finally:
        entry[0].release()
        with _locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                _locks.pop(url, None)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _topic_dicts(topics, text_len: int) -> list[dict]:
    out = []
    for t in topics:
        d = t.to_dict()
        start = t.char_start
        end = t.char_end if t.char_end is not None else text_len
        d["charStart"] = start
        d["charEnd"] = end
        d["chars"] = max(0, end - start) if start is not None else None
        out.append(d)
    return out


def text_status(text: str) -> str:
    n = len(text.strip())
    if n == 0:
        return "no_text"
    if n < LITTLE_TEXT_CHARS:
        return "little_text"
    return "ok"


def ingest(url: str) -> dict:
    with _one_at_a_time(url):
        tmp_path, ext = download_file(url)
        content_hash = _sha256(tmp_path)
        text = parse_file(tmp_path, ext)
        # A scan: its pages are read with text recognition (services/ocr.py),
        # once — a scan already indexed, or the same bytes uploaded again, has
        # its text rebuilt from the stored chunks instead.
        ocr = None
        topics_ext = ext
        if ext == ".pdf" and needs_ocr(tmp_path, text):
            topics_ext = ".md"  # the recognised text is Markdown, headings and all
            known = vector_store.stored_text(pdf_url=url) or vector_store.stored_text(content_hash=content_hash)
            if known:
                text, ocr = known, {"reused": True}
            else:
                try:
                    read = ocr_pdf(tmp_path)
                    text = read["text"]
                    ocr = {"pages_read": read["pages_read"], "pages_total": read["pages_total"]}
                except Exception:
                    traceback.print_exc()
                    ocr = {"failed": True}
        status = text_status(text)
        topics = extract_topics(tmp_path, topics_ext, text) if status != "no_text" else []
        result = {
            "status": "ok",
            "text_status": status,
            "content_hash": content_hash,
            "chars": len(text),
            "topics": _topic_dicts(topics, len(text)),
            "ocr": ocr,
        }
        if status == "no_text":
            result["chunks_stored"] = 0
            return result

        if vector_store.count_points(url) > 0:
            result.update(chunks_stored=0, reused="url")
            return result

        if vector_store.copy_points_for_hash(content_hash, url) > 0:
            result.update(chunks_stored=vector_store.count_points(url), reused="hash")
            return result

        chunks_with_offsets = chunk_text_with_offsets(text)
        chunks = [c for c, _ in chunks_with_offsets]
        chunk_topics = assign_topics_to_chunks(chunks_with_offsets, topics)
        embeddings = embed_texts(chunks)
        is_image = ext in IMAGE_EXTENSIONS
        vector_store.store_chunks(
            chunks,
            embeddings,
            url,
            topics=chunk_topics,
            metadata={"file_type": "image", "page": 1} if is_image else None,
            offsets=[o for _, o in chunks_with_offsets],
            content_hash=content_hash,
        )
        result["chunks_stored"] = len(chunks)
        return result
