# Combines the user's query + retrieved chunks and calls the LLM
# The chunks act as "context" so the LLM answers from the PDF, not its training

from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm(query: str, context_chunks: list[str]) -> str:
    # Merge all retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ]
    )
    return response.choices[0].message.content