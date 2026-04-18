from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

def ask_llm_stream(query: str, context_chunks: list[str]):
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a helpful assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}"
    )

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ],
        stream=True
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token