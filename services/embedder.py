# Converts each text chunk into a vector (list of floats)
# We use the SAME model at ingest time AND query time — this is critical

from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: list[str]) -> list[list[float]]:
    # OpenAI can embed multiple texts in one API call (batched)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    # Extract the vector for each text, in order
    return [item.embedding for item in response.data]

def embed_single(text: str) -> list[float]:
    return embed_texts([text])[0]