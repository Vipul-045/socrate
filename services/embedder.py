import cohere
from config import COHERE_API_KEY, EMBEDDING_MODEL

client = cohere.Client(api_key=COHERE_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="search_document"  # use this for ingesting
    )
    return response.embeddings

def embed_single(text: str) -> list[float]:
    response = client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="search_query"  # use this for querying
    )
    return response.embeddings[0]