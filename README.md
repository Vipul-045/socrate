**An autonomous learning platform that plans, teaches, tests, and adapts, so you don’t just study, you actually learn.**

## Run locally

Development never talks to the production Qdrant. With `SOCRATE_ENV=development` the service reads `.env.development` and never `.env`, and it refuses to start unless Qdrant is on this machine. Unset, the collection is `socrate_docs_dev`.

Anywhere else (production) `QDRANT_COLLECTION` must be set — production's is `socrate_docs` — and the service refuses to start without it or with a `*_dev` collection, so it can never quietly read an empty one. `render.yaml` lists every variable production needs.

1. Start Qdrant (and the backend's other stores) from `NChatAI/backend`: `docker compose -f docker-compose.dev.yml up -d --wait`.
2. Copy `.env.development.example` to `.env.development` and fill in the API keys.
3. Install and run:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
SOCRATE_ENV=development .venv/bin/uvicorn main:app --reload --port 8000
```

The backend's `.env.development` points `FileSearchURL` at `http://localhost:8000`.
