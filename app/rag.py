from pdf import load_pdf, split_text
from embeddings import get_embedding
from db import insert_document, search_similar
from models import ask_llm

def index_pdf(file_path):
    text = load_pdf(file_path)
    chunks = split_text(text)

    for chunk in chunks:
        emb = get_embedding(chunk)
        insert_document(chunk, emb)

    return "PDF indexed successfully"


def ask_question(question):
    q_emb = get_embedding(question)
    results = search_similar(q_emb)

    context = " ".join(results)
    answer = ask_llm(context, question)

    return answer