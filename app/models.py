from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm(context, question):
    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question: {question}
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content