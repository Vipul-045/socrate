# Wires everything together and starts the server

from fastapi import FastAPI
from routes.ingest import router as ingest_router
from routes.query  import router as query_router

app = FastAPI(title="Socrate Python Service")

app.include_router(ingest_router)
app.include_router(query_router)

# Run with: uvicorn main:app --reload --port 8000