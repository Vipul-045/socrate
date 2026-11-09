"""The topic finder for documents with no headings (topic_extractor
_extract_topics_ai): off unless TOPIC_AI_MODEL names a model, and then the
model named — never the Cohere API key — is what Cohere is asked for."""

import sys
import types

from services import topic_extractor

TEXT = "Cells release energy from glucose. " * 40


class FakeClient:
    calls: list = []

    def __init__(self, api_key=None):
        self.api_key = api_key

    def chat(self, message, model, temperature):
        FakeClient.calls.append({"model": model, "api_key": self.api_key})
        return types.SimpleNamespace(text='[{"title": "Energy from glucose", "level": 1, "snippet": "Cells release energy from glucose."}]')


def fake_cohere(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setitem(sys.modules, "cohere", types.SimpleNamespace(Client=FakeClient))


def test_off_without_a_model_nothing_is_called(monkeypatch):
    fake_cohere(monkeypatch)
    monkeypatch.setattr(topic_extractor, "TOPIC_AI_MODEL", None)
    topics = topic_extractor._extract_topics_ai(TEXT)
    assert FakeClient.calls == []
    assert [t.title for t in topics] == ["Document"]


def test_on_asks_for_the_named_model_not_the_key(monkeypatch):
    fake_cohere(monkeypatch)
    monkeypatch.setattr(topic_extractor, "TOPIC_AI_MODEL", "command-r7b-12-2024")
    monkeypatch.setattr(topic_extractor, "COHERE_API_KEY", "secret-key")
    topics = topic_extractor._extract_topics_ai(TEXT)
    assert FakeClient.calls == [{"model": "command-r7b-12-2024", "api_key": "secret-key"}]
    assert [t.title for t in topics] == ["Energy from glucose"]
