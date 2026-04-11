# No API key needed — runs locally on your machine for free

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once, ~80MB

def embed_texts(texts: list[str]) -> list[list[float]]:
    return model.encode(texts).tolist()

def embed_single(text: str) -> list[float]:
    return model.encode([text])[0].tolist()