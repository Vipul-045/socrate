from fastapi import FastAPI, UploadFile, File
import shutil
from rag import index_pdf, ask_question
from db import create_table

app = FastAPI()

create_table()

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": index_pdf(file_path)}

@app.get("/ask")
def ask(q: str):
    answer = ask_question(q)
    return {"answer": answer}