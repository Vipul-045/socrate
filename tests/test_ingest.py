"""A document is embedded and stored once (services/ingest.py), however many
times /process and /topic are called for it, and the same bytes uploaded again
reuse the stored vectors instead of calling the embedding API."""
import shutil
import pytest
from qdrant_client import QdrantClient

from services import ingest as ingest_mod, vector_store

FIXTURE = __file__.replace("test_ingest.py", "fixtures/chapter.pdf")


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_store, "client", QdrantClient(":memory:"))
    monkeypatch.setattr(vector_store, "_ensured", False)
    calls = {"embed": 0, "texts": 0}

    def fake_embed(texts):
        calls["embed"] += 1
        calls["texts"] += len(texts)
        return [[float(len(t) % 7), 1.0] + [0.0] * 1022 for t in texts]

    def fake_download(url):
        path = tmp_path / url.rsplit("/", 1)[-1]
        shutil.copy(FIXTURE if not url.endswith("blank.pdf") else BLANK, path)
        return str(path), ".pdf"

    def fake_ocr(path):
        calls["ocr"] = calls.get("ocr", 0) + 1
        if calls.get("ocr_fails"):
            raise RuntimeError("Gemini unavailable")
        return {"text": calls.get("ocr_text", ""), "pages_read": 1, "pages_total": 1}

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)
    monkeypatch.setattr(ingest_mod, "download_file", fake_download)
    monkeypatch.setattr(ingest_mod, "ocr_pdf", fake_ocr)
    return calls


BLANK = __file__.replace("test_ingest.py", "fixtures/blank.pdf")


def test_one_embedding_pass_however_often_it_is_asked(env):
    first = ingest_mod.ingest("http://s3.local/b/u1/chapter.pdf")
    again = ingest_mod.ingest("http://s3.local/b/u1/chapter.pdf")
    assert env["embed"] == 1
    assert first["chunks_stored"] > 0 and again.get("reused") == "url"
    assert vector_store.count_points("http://s3.local/b/u1/chapter.pdf") == first["chunks_stored"]
    assert [t["title"] for t in again["topics"]] == [t["title"] for t in first["topics"]]


def test_the_same_file_uploaded_again_reuses_its_vectors(env):
    first = ingest_mod.ingest("http://s3.local/b/u1/chapter.pdf")
    second = ingest_mod.ingest("http://s3.local/b/u2/chapter-copy.pdf")
    assert env["embed"] == 1
    assert second["reused"] == "hash"
    assert second["content_hash"] == first["content_hash"]
    assert vector_store.count_points("http://s3.local/b/u2/chapter-copy.pdf") == first["chunks_stored"]


def test_topics_carry_their_ranges_and_size(env):
    result = ingest_mod.ingest("http://s3.local/b/u1/chapter.pdf")
    assert result["topics"], "the fixture has headings"
    for t in result["topics"]:
        assert t["charStart"] is not None and t["charEnd"] >= t["charStart"]
        assert t["chars"] == t["charEnd"] - t["charStart"]


SCAN_TEXT = "# 1. Cells\n\n" + "Cells are the basic unit of life. " * 15 + "\n\n# 2. Energy\n\n" + "Cells release energy from glucose. " * 15


def test_a_scan_is_read_with_text_recognition_and_indexed(env):
    env["ocr_text"] = SCAN_TEXT
    result = ingest_mod.ingest("http://s3.local/b/u1/blank.pdf")
    assert env["ocr"] == 1
    assert result["text_status"] == "ok" and result["ocr"]["pages_read"] == 1
    assert result["chunks_stored"] > 0
    assert [t["title"] for t in result["topics"]] == ["1. Cells", "2. Energy"]


def test_a_scan_is_read_once(env):
    env["ocr_text"] = SCAN_TEXT
    first = ingest_mod.ingest("http://s3.local/b/u1/blank.pdf")
    again = ingest_mod.ingest("http://s3.local/b/u1/blank.pdf")
    copy = ingest_mod.ingest("http://s3.local/b/u2/blank.pdf")
    assert env["ocr"] == 1 and env["embed"] == 1
    assert again["ocr"] == {"reused": True} and copy["reused"] == "hash"
    assert [t["title"] for t in again["topics"]] == [t["title"] for t in first["topics"]]


def test_a_page_with_no_text_is_reported_not_indexed(env):
    result = ingest_mod.ingest("http://s3.local/b/u1/blank.pdf")
    assert result["text_status"] == "no_text"
    assert result["chunks_stored"] == 0 and env["embed"] == 0


def test_when_text_recognition_fails_the_document_is_reported_unread(env):
    env["ocr_fails"] = True
    result = ingest_mod.ingest("http://s3.local/b/u1/blank.pdf")
    assert result["text_status"] == "no_text" and result["ocr"] == {"failed": True}


def test_older_documents_stored_three_times_read_back_once(env):
    vector_store.ensure_collection()
    for _ in range(3):
        vector_store.store_chunks(["a", "b"], [[1.0] + [0.0] * 1023] * 2, "http://s3.local/b/old.pdf")
    assert vector_store.get_all_chunks("http://s3.local/b/old.pdf") == ["a", "b"]
