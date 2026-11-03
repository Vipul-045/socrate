"""The material a quiz, deck or exam is written from (/query/context): the
selected ranges' chunks, spread evenly across each range and sized to a
budget split by range size, each chunk once."""
import pytest
from qdrant_client import QdrantClient
from services import vector_store

URL = "http://s3.local/b/doc.pdf"
VEC = [1.0] + [0.0] * 1023


@pytest.fixture(autouse=True)
def store(monkeypatch):
    monkeypatch.setattr(vector_store, "client", QdrantClient(":memory:"))
    monkeypatch.setattr(vector_store, "_ensured", False)
    vector_store.ensure_collection()
    texts = [f"chunk{i:03d}".ljust(100, ".") for i in range(100)]  # 100 chunks of 100 chars
    vector_store.store_chunks(texts, [VEC] * 100, URL, offsets=[i * 100 for i in range(100)])


def test_whole_document_is_spread_evenly_not_cut_after_the_first_pages():
    out = vector_store.chunks_for_ranges(URL, [], budget_chars=2000)
    idx = [c["chunk_index"] for c in out["chunks"]]
    assert len(idx) == 20
    assert idx[0] < 10 and idx[-1] >= 90, idx  # reaches the end of the document


def test_ranges_get_budget_in_proportion_to_their_size():
    # range 0: chunks 0-9 (1,000 chars); range 1: chunks 50-89 (4,000 chars)
    out = vector_store.chunks_for_ranges(URL, [(0, 1000), (5000, 9000)], budget_chars=1000)
    r0 = [c for c in out["chunks"] if c["range"] == 0]
    r1 = [c for c in out["chunks"] if c["range"] == 1]
    assert len(r0) == 2 and len(r1) == 8
    assert all(0 <= c["char_start"] < 1000 for c in r0) and all(5000 <= c["char_start"] < 9000 for c in r1)
    assert out["ranges"] == [{"available_chars": 1000, "returned_chars": 200}, {"available_chars": 4000, "returned_chars": 800}]


def test_a_small_range_is_returned_whole():
    out = vector_store.chunks_for_ranges(URL, [(0, 300)], budget_chars=50000)
    assert [c["chunk_index"] for c in out["chunks"]] == [0, 1, 2]


def test_documents_stored_three_times_count_each_chunk_once():
    vector_store.store_chunks(["dup"] * 1, [VEC], URL, offsets=[0])  # a second copy of chunk 0's index
    out = vector_store.chunks_for_ranges(URL, [(0, 300)], budget_chars=50000)
    assert [c["chunk_index"] for c in out["chunks"]] == [0, 1, 2]


def test_without_offsets_ranges_cannot_be_applied():
    vector_store.store_chunks(["old"], [VEC], "http://s3.local/b/old.pdf")
    assert vector_store.chunks_for_ranges("http://s3.local/b/old.pdf", [(0, 10)], 1000)["located"] is False
